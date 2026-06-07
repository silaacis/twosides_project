"""
fingerprint_utils.py

Bu dosya, ilaçların SMILES gösterimlerinden Morgan Fingerprint
özellikleri çıkarmak için kullanılır.

Morgan Fingerprint, molekülün kimyasal alt yapılarını özetleyen
sayısal bir temsil yöntemidir. Bu projede GraphSAGE embedding'lerine
ek kimyasal bilgi sağlamak amacıyla kullanılacaktır.
"""

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )

    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return torch.tensor(array, dtype=torch.float)