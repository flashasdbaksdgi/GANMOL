# clusting of molecules based on scaffold

import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def DataFromSdfs(path):
    """
    Load data from multi sdf file
    """
    if os.path.exists(path):
        suppl = Chem.SDMolSupplier(path)
        return [x for x in suppl if x is not None]
    else:
        raise FileNotFoundError(f"{path} not found")


def GetScaffoldFromSmiles(mol_list):
    """
    Get scaffold from smiles
    """
    if not isinstance(mol_list, list):
        mol_list = [mol_list]

    smi_scaffolds = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        for mol in mol_list
    ]
    mol_scaffolds = [Chem.MolFromSmiles(smi_scaffold) for smi_scaffold in smi_scaffolds]
    return mol_scaffolds


def ScaffoldClusters(mol_list):
    scaffolds = {}
    clusters_list = []

    idx = 1
    for mol in mol_list:
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        if scaffold_smi not in scaffolds.keys():
            scaffolds[scaffold_smi] = idx
            idx += 1

        cluster_id = scaffolds[scaffold_smi]
        clusters_list.append(cluster_id)

    return clusters_list, scaffolds
