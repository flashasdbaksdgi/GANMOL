import vedo
from vedo.applications import SlicerPlotter
from vedo import show, Text2D
import numpy as np
from rdkit import Chem
import trimesh


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


def load_mol(filename):
    xyz = Chem.MolFromPDBFile(filename, removeHs=False)

    # print(Chem.MolToXYZBlock(xyz))
    # points = Chem.MolToXYZBlock(xyz)
    xyz = xyz_from_mol(xyz)
    # print(xyz)

    mesh = trimesh.convex.convex_hull(xyz)
    return mesh


def view_slicer(mesh):
    h2 = vedo.utils.trimesh2vedo(mesh)

    vol = vedo.volume.mesh2Volume(h2)
    # vol = vedo.volume.volumeFromMesh(h2)

    print(vol)
    plt = SlicerPlotter(
        vol,
        alpha=1,
        map2cells=False,
        draggable=True,
        bg="white",
        bg2="white",
        cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
        useSlider3D=False,
        title="Slice Visualizer",
        showIcon=True,
    )
    plt += Text2D("", font="arial")
    return plt.show().close()
    # return show(vol)


def load_stl(filename):
    """Loads triangular meshes from a file.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    meshes : list of :class:`~trimesh.base.Trimesh`
        The meshes loaded from the file.
    """
    # meshes = trimesh.load(filename)
    meshes = trimesh.exchange.load.load(filename, file_type="stl")
    # meshes = vedo.volume.interpolateToVolume(meshes)
    # mesh = trimesh.voxel.creation.voxelize(meshes, 0.5)
    # print(mesh)
    # If we got a scene, dump the meshes
    if isinstance(meshes, trimesh.Scene):
        meshes = list(meshes.dump())
        meshes = [g for g in meshes if isinstance(g, trimesh.Trimesh)]

    if isinstance(meshes, (list, tuple, set)):
        meshes = list(meshes)
        if len(meshes) == 0:
            raise ValueError("At least one mesh must be pmeshesent in file")
        for r in meshes:
            if not isinstance(r, trimesh.Trimesh):
                raise TypeError("Could not load meshes from file")
    elif isinstance(meshes, trimesh.Trimesh):
        return meshes
    else:
        raise ValueError("Unable to load mesh from file")


if __name__ == "__main__":

    xyz = "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/data/test/6nzp_protein.pdb"
    stl = "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/data/test/6nzp_protein.stl"
    stl1 = "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/data/test/t2.stl"
    # x = load_mol(xyz)
    x = load_stl(stl1)
    view_slicer(x)
