import vedo
from vedo.applications import SlicerPlotter
from vedo import show, Text2D
import numpy as np
from rdkit import Chem
import trimesh

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def xyz_from_mol(mol):
    """Extracts a numpy array of coordinates from a molecules.
    Returns a `(N, 3)` numpy array of 3d coords of given rdkit molecule
    Parameters
    ----------
    mol: rdkit Molecule
      Molecule to extract coordinates for
    Returns
    -------
    Numpy ndarray of shape `(N, 3)` where `N = mol.GetNumAtoms()`.
    """
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz

    # load xyz from sdf


def view_slicer(filename):
    xyz = Chem.MolFromPDBFile(filename, removeHs=False)

    # print(Chem.MolToXYZBlock(xyz))
    points = Chem.MolToXYZBlock(xyz)
    xyz = xyz_from_mol(xyz)
    # print(xyz)

    hull = trimesh.convex.convex_hull(xyz)
    h2 = vedo.utils.trimesh2vedo(hull)
    vol = vedo.volume.mesh2Volume(h2)
    print(vol)
    plt = SlicerPlotter(
        vol,
        bg="white",
        bg2="white",
        cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
        useSlider3D=False,
    )

    plt += Text2D("Molecule Slice Visualizer", font="arial")

    return plt.show().close()


def load_stl(path):
    mesh = trimesh.load(path)
    return mesh


if __name__ == "__main__":

    xyz = "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/data/test/6nzp_protein.pdb"

    view_slicer(xyz)
