from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torchdiffeq import odeint

from goflow.flow_matching.utils import (
    kabsch_align_batched,
    rmsd_loss,
    get_substruct_matches, 
    get_min_dmae_match_torch_batch,
)

from goflow.gotennet.models.components.outputs import Atomwise3DOut
from goflow.gotennet.models.representation.gotennet import GotenNet


class GraphTimeMLP(nn.Module):
    """
    MLP that takes per-graph times t_G and outputs per-graph coefficients.
    """
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t_G):
        # t_G: (num_graphs,) or (num_graphs, 1)
        if t_G.dim() == 1:
            t_G = t_G.unsqueeze(-1)
        coeffs = self.mlp(t_G)  # (num_graphs, 1)
        return coeffs.squeeze(-1)  # (num_graphs,)

def expand_coeffs_to_nodes(coeffs, batch):
    """
    Expand per-graph coefficients to per-node coefficients.
    Args:
        coeffs: (num_graphs,)
        batch: (num_nodes,) tensor mapping each node to its graph index
    Returns:
        per_node_coeffs: (num_nodes,)
    """
    return coeffs[batch]


class FlowModule(pl.LightningModule):
    def __init__(
            self,
            representation: GotenNet,
            lr: float = 5e-4,
            lr_decay: float = 0.5,
            lr_patience: int = 100,
            lr_minlr: float = 1e-6,
            lr_monitor: str = "validation/ema_val_loss",
            weight_decay: float = 0.01,
            num_steps: int = 10,
            num_samples: int = 1,
            seed: int = 1,
            output: Optional[Dict] = None,
            scheduler: Optional[Callable] = None,
            lr_warmup_steps: int = 0,
            use_ema: bool = False,
            sample_method: str = "gaussian",
            **kwargs
    ):
        super().__init__()
        self.representation = representation
        self.atomwise_3D_out_layer = Atomwise3DOut(n_in=representation.hidden_dim, n_hidden=output['n_hidden'], activation=F.silu)

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay

        self.num_steps = num_steps
        self.num_samples = num_samples

        self.use_ema = use_ema
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr

        self.seed = seed

        self.scheduler = scheduler
        self.sample_method = sample_method
        
        print("FM seed", self.seed)
        print(f"Sample method: {sample_method}")

        self.save_hyperparameters(ignore=['representation'])

        self.results_R = []  # save results in test_step

    def _sample_prior(self, x_1_N_3, batch):
        if self.sample_method == "gaussian":
            x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)
        elif self.sample_method == "pos_guess":
            x_0_N_3 = batch.pos_guess

        return x_0_N_3

    def get_perturbed_flow_point_and_time(self, batch: Data):
        x_1_N_3 = batch.pos
        x_0_N_3 = self._sample_prior(x_1_N_3, batch)
        
        t_G = torch.rand(batch.num_graphs, 1, device=self.device)
        t_N = t_G[batch.batch]

        x_1_aligned_N_3 = kabsch_align_batched(x_0_N_3, x_1_N_3, batch.batch)
        x_t_N_3 = (1 - t_N) * x_0_N_3 + t_N * x_1_aligned_N_3
        dx_dt_N_3 = x_1_aligned_N_3 - x_0_N_3

        return x_t_N_3, dx_dt_N_3, t_G

    def train_val_step(self, batch: Data) -> Tensor:
        x_t_N_3, dx_dt_N_3, t_G = self.get_perturbed_flow_point_and_time(batch)

        atom_N_3 = self.model_output(x_t_N_3, batch, t_G)

        return rmsd_loss(atom_N_3, dx_dt_N_3)

    def model_output(self, x_t_N_3, batch: Data, t_G: Tensor) -> Tensor:
        h_N_D, X_N_L_D = self.representation(x_t_N_3, t_G, batch)
        atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])
        return atom_N_3

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:        
        loss = self.train_val_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, 
                batch_size=batch.num_graphs)
        return loss


    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.train_val_step(batch)
        self.log("validation/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    # S... number of samples
    def test_step(self, batch: Batch, batch_idx: int):
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        def ode_func(t, x_t_N_3):
            t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
            model_forces_N_3 = self.model_output(x_t_N_3, batch, t_G)
            return model_forces_N_3

        # Generate num_samples trajectories for batch
        pos_gen_S_N_3 = torch.zeros((self.num_samples, batch.num_nodes, 3), device=self.device)
        
        for i in range(self.num_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            pos_init_N_3 = self._sample_prior(batch.pos, batch)
            pos_gen_S_N_3[i, ...] = odeint(ode_func, pos_init_N_3, t_T, method='euler')[-1]

        pos_gen_C_Nm_3 = []
        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            pos_gen_S_Nm_3 = pos_gen_S_N_3[:, mask]
            
            # If ground-truth pos exists, match and align samples to it
            if data.pos is not None:
                pos_gen_S_Nm_3 = self.substruct_match_and_kabsch_align_samples(data, pos_gen_S_Nm_3)

            # -------------------------- START: Aggregate the S samples --------------------------
            if self.num_samples > 1:
                pos_aggr_Nm_3 = torch.median(pos_gen_S_Nm_3, dim=0).values
                distances_S = torch.linalg.vector_norm(pos_gen_S_Nm_3 - pos_aggr_Nm_3, dim=(1, 2))
                pos_best_Nm_3 = pos_gen_S_Nm_3[torch.argmin(distances_S)]
            else:
                assert len(pos_gen_S_Nm_3) == 1
                pos_best_Nm_3 = pos_gen_S_Nm_3[0]
            # -------------------------- END: Aggregate the S samples --------------------------

            data.pos_gen = pos_best_Nm_3
            data.pos_gen_all_samples_S_N_3 = pos_gen_S_Nm_3
            pos_gen_C_Nm_3.append(pos_best_Nm_3)
            self.results_R.append(data.to("cpu"))
        
        return pos_gen_C_Nm_3
        
    def substruct_match_and_kabsch_align_samples(self, data, pos_gen_S_Nm_3):
        pos_gt_Nm_3 = data.pos
        
        # Substructure matching (batched for S)
        matches_M_N = get_substruct_matches(data.smiles)
        match_S_Nm = get_min_dmae_match_torch_batch(matches_M_N, pos_gt_Nm_3, pos_gen_S_Nm_3)
        pos_gen_S_Nm_3 = torch.gather(pos_gen_S_Nm_3, 1, match_S_Nm.unsqueeze(-1).expand(-1,-1,3))
        
        # Kabsch rotation
        S = pos_gen_S_Nm_3.shape[0]
        Nm = pos_gen_S_Nm_3.shape[1]
        
        # This is a trick to make the batched rotation to the GT molecule easy
        # Repeat GT pos S times (have to rotate each sample to it)
        pos_gt_SNm_3 = pos_gt_Nm_3.repeat(S, 1)
        pos_gen_SNm_3 = pos_gen_S_Nm_3.reshape(S*Nm, 3)
        
        batch = torch.arange(S, device=self.device).repeat_interleave(Nm)
        pos_gen_aligned_SNm_3 = kabsch_align_batched(pos_gt_SNm_3, pos_gen_SNm_3, batch)
        return pos_gen_aligned_SNm_3.reshape(S, Nm, 3)


    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        print("self.weight_decay", self.weight_decay)
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-7,
        )

        if self.scheduler and callable(self.scheduler):
            scheduler, _ = self.scheduler(optimizer=optimizer)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                min_lr=self.lr_minlr,
            )

        schedule = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [schedule]
