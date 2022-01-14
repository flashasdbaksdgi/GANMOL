import numpy as np
from openbabel.pybel import pybel


def pocket_density(mol, radius=3.0):
    """
    Calculate the density of a molecule in a pocket.
    """
    # Get the coordinates of the atoms in the pocket.
    pocket_coords = _get_pocket_coords(mol, radius)

    # Calculate the density of the pocket.
    return len(pocket_coords) / (4.0 * np.pi * radius ** 3)


def _get_pocket_coords(mol, radius):
    """
    Get the coordinates of the atoms in the pocket.
    """
    # Get the coordinates of the atoms in the pocket.
    pocket_coords = []
    for atom in mol.atoms:
        if _is_in_pocket(atom, mol, radius):
            pocket_coords.append(atom.coords)

    return pocket_coords


def _is_in_pocket(atom, mol, radius):
    """
    Determine if an atom is in the pocket.
    """
    # Get the coordinates of the atoms in the pocket.
    pocket_coords = _get_pocket_coords(mol, radius)

    # Determine if the atom is in the pocket.
    for pocket_coord in pocket_coords:
        if np.linalg.norm(atom.coords - pocket_coord) <= radius:
            return True

    return False


def pocket_identifier(protein):

    return
