import pickle
import rdkit.Chem.rdchem as rdchem

def create_organic_feat_dict(output_path='data/feat_dict_organic.pkl'):
    """
    Generates a feature dictionary tailored for Organic Chemistry 
    (C, H, N, O, S, P, Halogens).
    """
    
    def make_map(values):
        return {val: i for i, val in enumerate(values)}

    feat_dict = {}

    # 1. GetIsAromatic & IsInRing
    # ---------------------------------------------------------
    bool_vals = [False, True]
    feat_dict['GetIsAromatic'] = make_map(bool_vals)
    feat_dict['IsInRing'] = make_map(bool_vals)

    # 2. GetFormalCharge
    # ---------------------------------------------------------
    # 0: Standard
    # +1: Ammonium, Carbocations, Nitro N
    # -1: Carboxylates, Alkoxides, Nitro O
    # +/-2: Rare but possible in di-ions or zwitterions (e.g. sulfates)
    charges = [-2, -1, 0, 1, 2]
    feat_dict['GetFormalCharge'] = make_map(charges)

    # 3. GetHybridization
    # ---------------------------------------------------------
    # Standard Organic sets. 
    # SP3D/SP3D2 are included specifically for Hypervalent Sulfur (Sulfones) 
    # and Phosphorus (Phosphates/Wittig), which are common in organic sets.
    hyb_types = [
        rdchem.HybridizationType.UNSPECIFIED, # Fallback
        rdchem.HybridizationType.S,
        rdchem.HybridizationType.SP,          # Triple bonds / Allenes
        rdchem.HybridizationType.SP2,         # Double bonds / Aromatic
        rdchem.HybridizationType.SP3,         # Single bonds
        rdchem.HybridizationType.SP3D,        # Hypervalent P/S
        rdchem.HybridizationType.SP3D2        # Hypervalent S (e.g., SF6 derivatives)
    ]
    feat_dict['GetHybridization'] = make_map(hyb_types)

    # 4. GetTotalNumHs
    # ---------------------------------------------------------
    # Organic atoms rarely have > 4 Hydrogens attached (Methane/Ammonium = 4)
    # Range: 0 to 4
    feat_dict['GetTotalNumHs'] = make_map([0, 1, 2, 3, 4])

    # 5. GetTotalValence
    # ---------------------------------------------------------
    # Connectivity count (bond order sum).
    # Carbon max is 4. Nitrogen max 4/5. 
    # Sulfur/Phosphorus can go up to 6 or 7 (e.g. Perchloric acid, Sulfuric acid).
    # Range: 0 to 7 covers almost all organic functionality.
    feat_dict['GetTotalValence'] = make_map(list(range(8)))

    # 6. GetTotalDegree
    # ---------------------------------------------------------
    # Number of direct neighbors.
    # Carbon is max 4. P and S can be 5 or 6.
    # Range: 0 to 6
    feat_dict['GetTotalDegree'] = make_map(list(range(7)))

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(feat_dict, f)
    
    print(f"Organic feature dictionary saved to {output_path}")
    for k, v in feat_dict.items():
        print(f"  {k}: {len(v)} values {list(v.keys())}")

if __name__ == "__main__":
    create_organic_feat_dict()