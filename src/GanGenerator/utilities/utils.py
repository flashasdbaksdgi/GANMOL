import os
from glob import glob
import numpy as np
from openbabel import pybel


def centre_of_mass(mol: list) -> np.ndarray:
    """
    Calculate the centre of mass of a molecule.

    """
    mass = 0
    x = 0
    y = 0
    z = 0
    for atom in mol.atoms:
        if atom is not None:
            mass += atom.atomicmass
            x += atom.coords[0] * atom.atomicmass
            y += atom.coords[1] * atom.atomicmass
            z += atom.coords[2] * atom.atomicmass
    return np.array([x / mass, y / mass, z / mass])


def centroid_of_molecule(mol: list) -> np.array:
    """
    Calculate the centroid of a molecule.
    """
    x = 0
    y = 0
    z = 0
    for atom in mol.atoms:
        if atom is not None:
            x += atom.coords[0]
            y += atom.coords[1]
            z += atom.coords[2]
    return np.array([x / len(mol.atoms), y / len(mol.atoms), z / len(mol.atoms)])


def file_to_mol(filename, format=None):
    # TODO: check for openbabel molecule name?
    if format is None:
        format = filename.split(".")[-1]

    return next(pybel.readfile(format=format, filename=filename))


def file_search(type=None, target="*", specific=None):
    """searches files in sub dir
    Args:
        type (str, optional): Search file format
        target (str, optional): Identifier to search
        specific (str, optional): Specific folder to search
    Returns:
        list: Search result
    """
    BASE_DIR = os.getcwd()
    try:
        if specific is None:
            return sorted(glob(f"{BASE_DIR}/**/{target}.{type}", recursive=True))
        else:
            return sorted(
                glob(f"{BASE_DIR}/**/{specific}/{target}.{type}", recursive=True)
            )
    except Exception as error:
        print(f"{error} \n File not found anywhere.")
