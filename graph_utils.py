import torch
from torch_geometric.data import Data
from rdkit import Chem


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    atom_symbols = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
        'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
        'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
        'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
        'Hg', 'Pb', 'Unknown'
    ]

    features = (
        one_hot(atom.GetSymbol(), atom_symbols)
        + one_hot(atom.GetDegree(), list(range(11)))
        + one_hot(atom.GetTotalNumHs(), list(range(9)))
        + one_hot(atom.GetImplicitValence(), list(range(7)))
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_hot(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic(), atom.GetMass()]
    )

    return torch.tensor(features, dtype=torch.float)


def get_bond_features(bond):
    bond_type = bond.GetBondType()

    return torch.tensor(
        [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ],
        dtype=torch.float,
    )


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_feature = get_bond_features(bond)

        edge_indices.extend([[start, end], [end, start]])
        edge_attrs.extend([bond_feature, bond_feature])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)