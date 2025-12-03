import math
import torch
from torch import Tensor
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(species=mol2.species, coords=coords)
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def match_and_compute_rmsd(data):
    mol_pred = Molecule(
        species=data.atom_type.long().cpu().numpy(),
        coords=data.pos_gen.cpu().numpy(),
    )
    mol_ref = Molecule(
        species=data.atom_type.long().cpu().numpy(),
        coords=data.pos.cpu().numpy(),
    )
    try:
        rmsd = pymatgen_rmsd(
            mol_pred,
            mol_ref,
            ignore_chirality=True,
            threshold=0.5,
            same_order=False,
        )
    except Exception as e:
        print(f"Pymatgen failed with error: {e}")
        print(f"Shapes - Pred: {data.pos_gen.shape}, GT: {data.pos.shape}")
        pred_pos_N_3, gt_pos_N_3 = pred_atom_index_align(data.smiles, data.pos, data.pos_gen)
        rmsd = rmsd_loss(pred_pos_N_3, gt_pos_N_3)

    return rmsd


def compute_steric_clash_penalty(
    coords_N_3: torch.Tensor, r_threshold: float = 0.7, epsilon: float = 1.0
) -> torch.Tensor:
    """
    Compute a steric clash penalty based on a simplified LJ
    potential which only includes the repulsive 12-term. For any pair of atoms,
    if the distance r satisfies r < r_threshold then we add a penalty:

        V(r) = epsilon * [(r_threshold / r)^{12} - 1]    for r < r_threshold
        V(r) = 0                                          for r >= r_threshold

    The default r_threshold of 1.2 Å is chosen based on the observation that the
    shortest possible bond (C-C triple bond) is around this length.

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        r_threshold (float): Distance threshold below which a steric clash is 
                             penalized.
        epsilon (float): Scaling factor for the penalty.

    Returns:
        torch.Tensor: The total steric clash penalty for the entire set of atoms.
    """
    # Compute all pairwise distances
    dists_N_N = torch.cdist(coords_N_3, coords_N_3, p=2)
    # Consider only unique pairs (i < j)
    mask = torch.triu(torch.ones_like(dists_N_N, dtype=torch.bool), diagonal=1)
    dists_K = dists_N_N[mask]  # K = N*(N-1)/2

    # Identify clashes where the distance is below the threshold.
    clash_mask = dists_K < r_threshold
    penalty_K = torch.zeros_like(dists_K)
    
    penalty_K[clash_mask] = epsilon * ((r_threshold / dists_K[clash_mask]) ** 12 - 1)
    total_penalty = penalty_K.sum()
    return total_penalty


def rmsd_loss(pred_N_3: Tensor, gt_N_3: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((pred_N_3 - gt_N_3) ** 2))


def kabsch_align_batched(x_0_N_3, x_1_N_3, batch):
    # x_0_N_3, x_1_N_3 are tensors of shape (N, 3)
    # batch is a 1D tensor of length N with group indices.
    device = x_0_N_3.device
    Nm = int(batch.max().item() + 1)

    # Compute counts and centers
    counts = torch.bincount(batch, minlength=Nm).to(x_0_N_3.dtype).clamp(min=1)
    
    # Compute group centroids
    centers_x0_Nm_3 = torch.zeros((Nm, 3), dtype=x_0_N_3.dtype, device=device)
    centers_x1_Nm_3 = torch.zeros((Nm, 3), dtype=x_1_N_3.dtype, device=device)
    centers_x0_Nm_3.index_add_(0, batch, x_0_N_3)
    centers_x1_Nm_3.index_add_(0, batch, x_1_N_3)
    centers_x0_Nm_3 = centers_x0_Nm_3 / counts.unsqueeze(1)
    centers_x1_Nm_3 = centers_x1_Nm_3 / counts.unsqueeze(1)

    # Center the points
    x0_centered_N_3 = x_0_N_3 - centers_x0_Nm_3[batch]
    x1_centered_N_3 = x_1_N_3 - centers_x1_Nm_3[batch]

    # Covariance Matrix construction
    prod_N_3_3 = x1_centered_N_3.unsqueeze(2) * x0_centered_N_3.unsqueeze(1)
    M_Nm_3_3 = torch.zeros((Nm, 3, 3), dtype=prod_N_3_3.dtype, device=device)
    M_Nm_3_3.index_add_(0, batch, prod_N_3_3)

    # Batched SVD
    U_Nm_3_3, _, Vt_Nm_3_3 = torch.linalg.svd(M_Nm_3_3)

    # 1. Compute determinant of the uncorrected rotation matrix UV^T
    # use the property: det(UV^T) = det(U) * det(V^T)
    R_temp = torch.bmm(U_Nm_3_3, Vt_Nm_3_3)
    det_Nm = torch.det(R_temp)
    
    # 2. Reflection Correction:
    # Instead of constructing a diagonal matrix D and doing R = U @ D @ Vt,
    # flip the sign of the last row of Vt where det < 0.
    mask_neg = det_Nm < 0
    if mask_neg.any():
        # Clone to avoid in-place modification issues if gradients are required later
        Vt_Nm_3_3 = Vt_Nm_3_3.clone() 
        Vt_Nm_3_3[mask_neg, 2, :] *= -1

    # 3. Final Rotation
    R_opt_Nm_3_3 = torch.bmm(U_Nm_3_3, Vt_Nm_3_3)

    # Apply rotation
    # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
    x_1_rotated_N_3 = torch.bmm(x1_centered_N_3.unsqueeze(1), R_opt_Nm_3_3[batch]).squeeze(1)

    return x_1_rotated_N_3 + centers_x0_Nm_3[batch]


@torch.no_grad()
def get_shortest_path_x_1(
    x_target_N_3: torch.Tensor,  # e.g., x0: (N,3) — fixed target
    x_moving_N_3: torch.Tensor,  # e.g., x1: (N,3) — will be rotated/translated
    return_aligned: bool = True
):
    """
    Kabsch alignment: find R,t that minimize || (x_moving R + t) - x_target ||_F.
    Returns R (3x3), t (3,), rmsd (scalar), and optionally the aligned coords.
    """
    assert x_moving_N_3.shape == x_target_N_3.shape and x_moving_N_3.shape[-1] == 3

    # 1) center both point sets
    c_moving_1_3 = x_moving_N_3.mean(dim=0, keepdim=True)  # (1,3)
    c_target_1_3 = x_target_N_3.mean(dim=0, keepdim=True)  # (1,3)
    X = x_moving_N_3 - c_moving_1_3                          # (N,3)
    Y = x_target_N_3 - c_target_1_3                          # (N,3)

    # 2) covariance and SVD
    # Move "moving" onto "target": M = X^T Y
    M_3_3 = X.transpose(0, 1) @ Y                            # (3,3)
    U, S, Vt = torch.linalg.svd(M_3_3, full_matrices=False)  # U (3,3), Vt (3,3)

    # 3) rotation (proper, no reflection)
    V = Vt.transpose(0, 1)
    R_3_3 = V @ U.transpose(0, 1)                            # (3,3)
    if torch.det(R_3_3) < 0:
        # flip last column of V (== last row of Vt) and recompute
        V[:, -1] *= -1
        R_3_3 = V @ U.transpose(0, 1)

    # 4) translation so that (x_moving R + t) best matches x_target
    # Using row-vector convention: x' = x R + t
    t_3 = (c_target_1_3 - c_moving_1_3 @ R_3_3).squeeze(0)   # (3,)

    # 5) rmsd
    if return_aligned:
        x_aligned_N_3 = x_moving_N_3 @ R_3_3 + t_3           # (N,3)
        rmsd = torch.sqrt(torch.mean((x_aligned_N_3 - x_target_N_3) ** 2))
        #return R_3_3, t_3, rmsd, x_aligned_N_3
        return x_aligned_N_3
    else:
        # Equivalent RMSD via centered coords
        X_rot = X @ R_3_3
        rmsd = torch.sqrt(torch.mean((X_rot - Y) ** 2))
        return R_3_3, t_3, rmsd
        

def get_min_rmsd_match(matches, gt_pos, pred_pos):
    rmsd_M = []
    for match in matches:
        pred_pos_match = pred_pos[list(match)]
        gt_pos_aligned = get_shortest_path_x_1(pred_pos_match, gt_pos)
        rmsd_M.append(rmsd_loss(pred_pos_match, gt_pos_aligned))
    return list(matches[rmsd_M.index(min(rmsd_M))])


def calc_DMAE(dm_ref, dm_guess, mape=False):
    if mape:
        retval = abs(dm_ref - dm_guess) / dm_ref
    else:
        retval = abs(dm_ref - dm_guess)
    return np.triu(retval, k=1).sum() / len(dm_ref) / (len(dm_ref) - 1) * 2


def get_min_dmae_match(matches, ref_pos, prb_pos):
    dmaes = []
    for match in matches:
        match_pos = prb_pos[list(match)]
        dmae = calc_DMAE(cdist(ref_pos, ref_pos), cdist(match_pos, match_pos))
        dmaes.append(dmae)
    return list(matches[dmaes.index(min(dmaes))])


def get_substruct_matches(smarts):
    smarts_r, smarts_p = smarts.split(">>")
    mol_r = Chem.MolFromSmarts(smarts_r)
    mol_p = Chem.MolFromSmarts(smarts_p)

    matches_r = list(mol_r.GetSubstructMatches(mol_r, uniquify=False, useChirality=True))
    map_r = np.array([atom.GetAtomMapNum() for atom in mol_r.GetAtoms()]) - 1
    map_r_inv = np.argsort(map_r)
    for i in range(len(matches_r)):
        matches_r[i] = tuple(map_r[np.array(matches_r[i])[map_r_inv]])

    matches_p = list(mol_p.GetSubstructMatches(mol_p, uniquify=False, useChirality=True))
    map_p = np.array([atom.GetAtomMapNum() for atom in mol_p.GetAtoms()]) - 1
    map_p_inv = np.argsort(map_p)
    for i in range(len(matches_p)):
        matches_p[i] = tuple(map_p[np.array(matches_p[i])[map_p_inv]])

    matches = set(matches_r) & set(matches_p)
    matches = list(matches)
    matches.sort()
    return matches


def get_min_dmae_match_torch_batch(matches_M_N, pos_gt_N_3, pos_pred_S_N_3):
    """
    Given a set of matches (each a tuple of indices), ground-truth positions (pos_gt_N_3),
    and S samples of predicted positions (pos_pred_S_N_3), compute the DMAE for each match
    in each sample and return the match (as a list) with the minimal DMAE per sample.

    Args:
        matches_M_N: list or tensor of candidate matches of shape (M, N)
                     where M is the number of candidate matches and N is the number of atoms.
        pos_gt_N_3:  tensor of ground-truth atom positions of shape (N, 3).
        pos_pred_S_N_3: tensor of predicted atom positions for S samples (shape: S, N_total, 3)
                        where N_total must be large enough to index by matches_M_N.
    Returns:
        A list (or tensor) of best candidate match indices for each sample, of shape (S, N).
        Each row corresponds to the candidate match (from matches_M_N) that minimizes the DMAE
        for that sample.
    """
    matches_M_N = torch.as_tensor(matches_M_N, dtype=torch.long, device=pos_pred_S_N_3.device)

    # 1. Select and Permute Positions
    # Shape: (S, N_total, 3) -> (S, M, N, 3)
    # This gathers the specific atoms defined in matches_M_N for every sample
    candidate_pred_pos_S_M_N_3 = pos_pred_S_N_3[:, matches_M_N]

    S, M, N, _ = candidate_pred_pos_S_M_N_3.shape

    # 2. Batched Distance Matrix Calculation
    # Reshape to (S*M, N, 3) for batched cdist
    flat_pred_pos_SM_N_3 = candidate_pred_pos_S_M_N_3.reshape(S * M, N, 3)
    d_matches_SM_N_N = torch.cdist(flat_pred_pos_SM_N_3, flat_pred_pos_SM_N_3)

    # 3. Reference Distance Matrix
    # Shape: (N, N)
    d_ref_N_N = torch.cdist(pos_gt_N_3.unsqueeze(0), pos_gt_N_3.unsqueeze(0)).squeeze(0)

    # 4. Compute Absolute Difference
    # Broadcasting: (SM, N, N) - (1, N, N)
    diff_SM_N_N = torch.abs(d_matches_SM_N_N - d_ref_N_N.unsqueeze(0))

    # 5. Calculate DMAE
    # Sum over the last two dimensions (N, N)
    dmaes_SM = diff_SM_N_N.sum(dim=(-1, -2)) / (N * (N - 1))

    # 6. Find Best Match
    dmaes_S_M = dmaes_SM.view(S, M)
    best_idx_S = torch.argmin(dmaes_S_M, dim=1)

    return matches_M_N[best_idx_S]

def pred_atom_index_align(smiles, gt_atom_pos, pred_atom_pos):
    matches = get_substruct_matches(smiles)
    match = get_min_rmsd_match(matches, gt_atom_pos, pred_atom_pos)

    pred_atom_pos_match = pred_atom_pos[match]
    gt_atom_pos_aligned = get_shortest_path_x_1(pred_atom_pos_match, gt_atom_pos)

    return pred_atom_pos_match, gt_atom_pos_aligned


def pred_atom_index_align_mad(smiles, gt_atom_pos, pred_atom_pos) -> Tensor:
    matches = get_substruct_matches(smiles)
    match = get_min_dmae_match(matches, gt_atom_pos, pred_atom_pos)
    return pred_atom_pos[match]

def calc_DMAE_torch(dm_ref, dm_guess, mape=False):
    """
    Compute the Distance Matrix Absolute Error (DMAE) between two distance matrices.
    dm_ref and dm_guess are torch tensors of shape (N, N).
    """
    if mape:
        diff = torch.abs(dm_ref - dm_guess) / dm_ref
    else:
        diff = torch.abs(dm_ref - dm_guess)
    # Keep only the upper triangle (excluding the diagonal)
    diff_upper = torch.triu(diff, diagonal=1)
    N = dm_ref.shape[0]
    return 2 * diff_upper.sum() / (N * (N - 1))
