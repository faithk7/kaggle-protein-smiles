from rdkit import Chem


def normalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    normalized_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return normalized_smiles
