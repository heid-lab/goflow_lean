import pickle
import os
import numpy as np
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT
from ase.io import iread
from typing import List, Dict, Optional

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Constants
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}

# --- Helper Functions ---
def parse_xyz_corpus_ase(filename):
    return [atoms.positions for atoms in iread(filename)]

def get_closest_value(v: Dict, feat):
    if feat in v: return v[feat]
    return v[min(v.keys(), key=lambda k: abs(int(k) - int(feat)))]

def get_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)

def process_state(smiles: str, feat_dict: Dict) -> SimpleNamespace:
    """
    Processes a single molecule state (Reactant OR Product)
    Returns a namespace containing the mol, features, permutations, and adjacency.
    """
    # 1. Setup
    mol = get_mol(smiles)
    
    # 2. Calc Permutations (MapNum -> Index mappings)
    # perm: map_num for atom at index i
    perm = np.array([a.GetAtomMapNum() for a in mol.GetAtoms()]) - 1
    # perm_inv: index of atom with map_num i (Canonical ordering)
    perm_inv = np.argsort(perm)

    # 3. Extract Node Features
    atoms = np.array(mol.GetAtoms())[perm_inv]
    z = [atom.GetAtomicNum() for atom in atoms]
    
    feat_indices = []
    for atom in atoms:
        atomic_feat = []
        for k, v in feat_dict.items():
            atomic_feat.append(get_closest_value(v, getattr(atom, k)()))
        feat_indices.append(atomic_feat)
    
    # One-Hot Encoding
    feat_tensor = torch.tensor(feat_indices, dtype=torch.long)
    num_cls = [len(v) for k, v in feat_dict.items()]
    feat_onehot = [F.one_hot(feat_tensor[:, i], num_classes=n) for i, n in enumerate(num_cls)]
    final_feat = torch.cat(feat_onehot, dim=-1).float()

    # 4. Adjacency Matrix (Reordered to Canonical)
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    adj_perm = adj[perm_inv, :][:, perm_inv]

    return SimpleNamespace(
        mol=mol,
        z=torch.tensor(z, dtype=torch.long),
        feat=final_feat,
        perm=torch.from_numpy(perm),
        perm_inv=torch.from_numpy(perm_inv),
        adj_perm=adj_perm
    )

def extract_bond_types(state: SimpleNamespace, row: np.ndarray, col: np.ndarray) -> torch.Tensor:
    """Extracts bond types for specific edges defined by row/col in the canonical graph."""
    bond_types = []
    # Map canonical indices back to local atom indices
    atom_i = state.perm_inv[row]
    atom_j = state.perm_inv[col]

    for i, j in zip(atom_i, atom_j):
        b = state.mol.GetBondBetweenAtoms(int(i), int(j))
        bond_types.append(BOND_TYPES[b.GetBondType()] if b else 0)
        
    return torch.tensor(bond_types, dtype=torch.long)

# --- Main Logic ---

def generate_graph_data(
    r_smiles: str,
    p_smiles: str,
    pos_guess: torch.Tensor,
    pos_gt: torch.Tensor,
    feat_dict: Dict,
    data_cls=Data
):
    # 1. Process Reactant and Product Independently
    R = process_state(r_smiles, feat_dict)
    P = process_state(p_smiles, feat_dict)

    # Sanity Checks
    assert torch.equal(R.z, P.z), "Atomic number mismatch between R and P"
    N = len(R.z)
    if pos_gt is not None:
        assert len(pos_gt) == N
    if pos_guess is not None:
        assert len(pos_guess) == N

    # 2. Combine Graphs (Union of Edges)
    adj = R.adj_perm + P.adj_perm
    row, col = adj.nonzero()
    
    # 3. Extract Edge Features based on Union
    r_edge_type = extract_bond_types(R, row, col)
    p_edge_type = extract_bond_types(P, row, col)

    # Sort edges (PyG convention: row-major sort)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    
    edge_index = edge_index[:, perm]
    edge_type_final = (r_edge_type[perm] * len(BOND_TYPES)) + p_edge_type[perm]

    # 4. Return Data
    return data_cls(
        num_nodes=N,
        atom_type=R.z,
        r_feat=R.feat,
        p_feat=P.feat,
        pos=torch.from_numpy(pos_gt).float() if isinstance(pos_gt, np.ndarray) else pos_gt,
        pos_guess=torch.from_numpy(pos_guess).float() if isinstance(pos_guess, np.ndarray) else pos_guess,
        edge_index=edge_index,
        edge_type=edge_type_final,
        smiles=f"{r_smiles}>>{p_smiles}",
    )

def process_reaction_data(
    feat_dict: Dict,
    rxn_smiles: str,
    rxn_id: int,
    guess_xyzs_C_N_3: Optional[List[np.ndarray]] = None,
    gt_xyzs_C_N_3: Optional[List[np.ndarray]] = None,
):
    r_smi, p_smi = rxn_smiles.split(">>")
    data_list = []
    
    assert guess_xyzs_C_N_3 is not None or gt_xyzs_C_N_3 is not None
    if guess_xyzs_C_N_3 is None: guess_xyzs_C_N_3 = [None] * len(gt_xyzs_C_N_3)
    if gt_xyzs_C_N_3 is None: gt_xyzs_C_N_3 = [None] * len(guess_xyzs_C_N_3)

    for i, (guess_N_3, gt_N_3) in enumerate(zip(guess_xyzs_C_N_3, gt_xyzs_C_N_3)):
        try:
            data = generate_graph_data(r_smi, p_smi, guess_N_3, gt_N_3, feat_dict)
            data.rxn_index = rxn_id
            data_list.append(data)
        except Exception as e:
            print(f"!!! Skipping rxn id {rxn_id} mol {i}: {e} !!!")

    return data_list


if __name__ == "__main__":
    """
    Process RDB7 dataset.
    """
    import pandas as pd
    import argparse
    import tqdm

    with open("data/RDB7/feat_dict_organic.pkl", "rb") as f:
        feat_dict_organic = pickle.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to rdb7_full.csv")
    parser.add_argument("--xyz_file", type=str, required=True, help="Path to rdb7_full.xyz")
    parser.add_argument("--save_filepath", type=str, default="data/processed")
    args = parser.parse_args()
    
    # Read CSV file with reaction data
    df = pd.read_csv(args.csv_file)
    rxn_smiles_R = df.smiles
    rxn_indices_R = df.rxn
    # Read xyz file with gt TS data
    xyz_blocks_3R = parse_xyz_corpus_ase(args.xyz_file)
    rxn_block_R_N_3 = [np.array(xyz_blocks_3R[i+1]) for i in range(0, len(xyz_blocks_3R), 3)] # i+1 => take the TS

    data_list_R = []
    for (id, smiles, xyz_N_3) in tqdm.tqdm(zip(rxn_indices_R, rxn_smiles_R, rxn_block_R_N_3)):
        data_1 = process_reaction_data(feat_dict_organic, smiles, id, guess_xyzs_C_N_3=None, gt_xyzs_C_N_3=xyz_N_3[None, ...])
        data_list_R.extend(data_1)

    os.makedirs(os.path.dirname(args.save_filepath), exist_ok=True)
    with open(args.save_filepath, "wb") as f:
        pickle.dump(data_list_R, f)
        