# scaffold generations and similarity calculations

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork, MurckoScaffold
from rdkit import DataStructs


def similarity(mol1, mol2, metric="tanimoto"):
    """
    Calculate the similarity between two molecules
    """
    available_metrics = [
        "tanimoto",
        "dice",
        "tversky",
        "Sokal",
        "cosine",
        "Kulczynski",
        "McConnaughey",
    ]
    if metric not in available_metrics:
        raise ValueError(f"Unknown metric: {metric}. Metric not available")
    molecules = [mol1, mol2]

    if not all(isinstance(x, Chem.Mol) for x in molecules):
        raise ValueError("molecules must be a rdkit.Chem.Mol object")

    # get fingerprint
    mol1, mol2 = [Chem.RDKFingerprint(x) for x in molecules]

    # calculate similarity
    if metric == "tanimoto":
        return DataStructs.TanimotoSimilarity(mol1, mol2)
    elif metric == "dice":
        return DataStructs.DiceSimilarity(mol1, mol2)
    elif metric == "tversky":
        return DataStructs.TverskySimilarity(mol1, mol2, 0.5, 0.5)
    elif metric == "Sokal":
        return DataStructs.SokalSimilarity(mol1, mol2)
    elif metric == "cosine":
        return DataStructs.CosineSimilarity(mol1, mol2)
    elif metric == "Kulczynski":
        return DataStructs.KulczynskiSimilarity(mol1, mol2)
    else:
        return DataStructs.McConnaugheySimilarity(mol1, mol2)


def smi2mol(smile, scaffold=False):
    """
    Convert a SMILES string to a rdkit.Chem.Mol object
    """
    if scaffold:
        smile = MurckoScaffold.MurckoScaffoldSmiles(smile, includeChirality=True)

    return Chem.MolFromSmiles(smile)


if __name__ == "__main__":
    smi1 = "c1cnc2cc(Nc3ccn3)nc(NC3CC4CCC(C3)N4)c2c1"
    smi2 = "c1ccc(CNc2cc(-c3cnc3)cc(Nc3cnccn3)n2)cc1"
    available_metrics = [
        "tanimoto",
        "dice",
        "tversky",
        "Sokal",
        "cosine",
        "Kulczynski",
        "McConnaughey",
    ]
    x = "Algorithms\tNormal Similaity\tScaffold Similarity"
    print(x)
    print("-----------------------------------------------------------")

    for i in available_metrics:
        print(
            i.capitalize().ljust(15, " "),
            ":",
            similarity(
                smi2mol(smi1, scaffold=False), smi2mol(smi2, scaffold=False), metric=i
            ),
            "\t",
            similarity(
                smi2mol(smi1, scaffold=True), smi2mol(smi2, scaffold=True), metric=i
            ),
        )
