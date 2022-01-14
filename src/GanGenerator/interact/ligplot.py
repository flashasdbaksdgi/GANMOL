"""
LigPlot
-------------------------------------------------------------------------------
A class to view 2D and 3D interaction map of protein and ligand.
"""
__author__      = "Rahul Brahma"

from rdkit import Chem
import os
from rdkit import Geometry
import prolif
import py3Dmol
import MDAnalysis as mda
import prolif as plf
from prolif.plotting.network import LigNetwork
from rich.console import Console
#import imgkit
console = Console()

class LigPlot:


    def __init__(self,*args,**kwargs):
       self.protein= kwargs.get('protein', None)
       self.ligand = kwargs.get('ligand', None)
       self.BASE_DIR = kwargs.get('BASE_DIR', os.getcwd())

    def __prep_protein(self, protein=None):
        # load protein
        prot = Chem.MolFromPDBFile(protein, removeHs=False)
        #prot = Chem.AddHs(prot)
        prot = plf.Molecule(prot)
        #protein_name = os.path.basename(protein)
        #console.print(f"{protein_name} has {prot.n_residues} of residues")
        return prot

    def __prep_ligand(self, ligand):
        #load ligand and prep
        *_, file_format = ligand.rsplit(".")
        if file_format == 'sdf':
            lig_suppl =  list(plf.sdf_supplier(ligand))
        elif file_format == 'pdb':
            """ FIXME: valence error in sdf file conversion from pdb by rdkit
            Avoiding sanitization.
            """
            try:
                lig_suppl = Chem.MolFromPDBFile(
                    ligand, removeHs=False, sanitize=False)
                self.temp_file = os.path.join(self.BASE_DIR, 'temp_temp.pdb')
                try:
                    Chem.SanitizeMol(lig_suppl, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    Chem.rdmolops.SanitizeFlags.SANITIZE_NONE
                except Exception as er:
                    #print(er)
                    pass
                Chem.MolToMolFile(lig_suppl, self.temp_file,kekulize=True)
                lig_suppl = list(plf.sdf_supplier(self.temp_file))

            except AttributeError as er:
                #raise AttributeError(f"Enter ligand file is not a valid structure. \n {er}")
                ref = mda.Universe(ligand)
                ref = plf.Molecule.from_mda(ref)
                lig_suppl = [ref]

        elif file_format == 'mol2':
            lig_suppl = list(plf.mol2_supplier(ligand))
        return lig_suppl

    def fingerprint(self, protein_plf, ligand_plf, verbose=False,  return_atoms=False):
        """Calculates fingerprint of interaction of protein and ligand

        Args:
            protein_plf (proLIF): rdkit like mool object
            ligand_plf (proLIF): rdkit like mol object
            verbose (bool, optional): Prints out all the fingerprint information avaiable. Defaults to False.

        Returns:
            DataFrame: All the fingerprint information
        """
        
        if not isinstance(protein_plf, prolif.molecule.Molecule):
            print("Warning")
            protein_plf = self.__prep_protein(protein_plf)
        if not isinstance(ligand_plf,  list):
            ligand_plf = self.__prep_ligand(ligand_plf)
            
            
        # get fingerprint
        finger_print = plf.Fingerprint()
        
        finger_print.run_from_iterable(ligand_plf, protein_plf)
        #print(ligand_plf)
        #if isinstance (ligand_plf, list):
        #    l = ligand_plf
        #    ligand_plf= ligand_plf[0]
        #finger_print = plf.Fingerprint().run(l,ligand_plf, protein_plf)    
        #finger_print.generate(ligand_plf, protein_plf)
        # get lig-prot interactions with atom info
        self.fp_df =finger_print.to_dataframe(return_atoms=True)
        #fp_df.T
        if verbose:
            console.print(self.fp_df.T)
        return self.fp_df


    def get_ring_centroid(self, mol, index):
        """Calculate center of ring given mol

        Args:
            mol (rdkit mol): RDKIT like mol of proLIF
            index (int): Index of atom in ring

        Raises:
            ValueError: If invalid or mol without ring information will raise ValueError.

        Returns:
            proLIF obj: Coordinates of center of ring found
        """
        # find ring using the atom index
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        ri = mol.GetRingInfo()
        for r in ri.AtomRings():
            if index in r:
                break
        else:
            raise ValueError("No ring containing this atom index was found in the given molecule")
        # get centroid
        coords = mol.xyz[list(r)]
        ctd = plf.utils.get_centroid(coords)
        return Geometry.Point3D(*ctd)



    def __view_3D(self, df, ligand_mol, protein_mol, color=None, save=False):

        colors = color or {
            "HBAcceptor": "blue",
            "HBDonor": "red",
            "Cationic": "green",
            "PiStacking": "purple",
            "Hydrophobic": "yellow",
            "Anionic": "cyan",

        }

        # JavaScript functions
        resid_hover = """function(atom,viewer) {{
            if(!atom.label) {{
                atom.label = viewer.addLabel('{0}:'+atom.atom+atom.serial,
                    {{position: atom, backgroundColor: 'mintcream', fontColor:'black'}});
            }}
        }}"""
        hover_func = """
        function(atom,viewer) {
            if(!atom.label) {
                atom.label = viewer.addLabel(atom.interaction,
                    {position: atom, backgroundColor: 'black', fontColor:'white'});
            }
        }"""
        unhover_func = """
        function(atom,viewer) {
            if(atom.label) {
                viewer.removeLabel(atom.label);
                delete atom.label;
            }
        }"""

        v = py3Dmol.view(800, 600)
        v.removeAllModels()

        INTERACTION_CLASS = ["PiStacking",
                             "EdgeToFace", "FaceToFace", "PiCation"]

        models = {}
        mid = -1
        for i, row in df.T.iterrows():
            lresid, presid, interaction = i
            lindex, pindex = row[0]
            lres = ligand_mol[lresid]
            pres = protein_mol[presid]
            # set model ids for reusing later
            for resid, res, style in [(lresid, lres, {"colorscheme": "cyanCarbon"}),
                                      (presid, pres, {})]:
                if resid not in models.keys():
                    mid += 1
                    v.addModel(Chem.MolToMolBlock(res), "sdf")
                    model = v.getModel()
                    model.setStyle({}, {"stick": style})
                    # add residue label
                    model.setHoverable({}, True, resid_hover.format(resid), unhover_func)
                    models[resid] = mid
            # get coordinates for both points of the interaction
            if interaction in INTERACTION_CLASS:
                p1 = self.get_ring_centroid(lres, lindex)
            else:
                p1 = lres.GetConformer().GetAtomPosition(lindex)
            if interaction in INTERACTION_CLASS:
                p2 = self.get_ring_centroid(pres, pindex)
            else:
                p2 = pres.GetConformer().GetAtomPosition(pindex)
            # add interaction line
            v.addCylinder({"start": dict(x=p1.x, y=p1.y, z=p1.z),
                           "end":   dict(x=p2.x, y=p2.y, z=p2.z),
                           "color": colors[interaction],
                           "radius": .15,
                           "dashed": True,
                           "fromCap": 1,
                           "toCap": 1,
                          })
            # add label when hovering the middle of the dashed line by adding a dummy atom
            c = Geometry.Point3D(*plf.utils.get_centroid([p1, p2]))
            modelID = models[lresid]
            model = v.getModel(modelID)
            model.addAtoms([{"elem": 'Z',
                             "x": c.x, "y": c.y, "z": c.z,
                             "interaction": interaction}])
            model.setStyle({"interaction": interaction}, {"clicksphere": {"radius": .5}})
            model.setHoverable(
                {"interaction": interaction}, True,
                hover_func, unhover_func)

        pdb = Chem.MolToPDBBlock(protein_mol, flavor=0x20 | 0x10)
        v.addModel(pdb, "pdb")
        model = v.getModel()
        model.setStyle({}, {"cartoon": {"style":"edged"}})
        v.zoomTo({"model": list(models.values())})
        # no support for image by py3dmol.
        #if save:
        #    with open("test.png", "w+") as im:
        #        print(v.png(), file=im)
        return v.show()


    def LigPlot3D(self,protein=None, ligand=None, verbose=False, save=False):
        if protein is None and self.protein is not None:
            protein = self.protein
        prot = self.__prep_protein(protein)
        if ligand is None and self.ligand is not None:
            ligand = self.ligand
        lig_suppl = self.__prep_ligand(ligand)
        df = self.fingerprint(prot, lig_suppl, verbose=verbose)
        return self.__view_3D(df, lig_suppl[0], prot, save=save)


    def __view_2D(self, fp_df, ligand_mol, save=False, html=False):
        net = LigNetwork.from_ifp(fp_df, ligand_mol,
                                  # replace with `kind="frame", frame=0` for the other depiction
                                  kind="aggregate", threshold=.3,
                                  rotation=270)
        if save:
            # prolif save image sin html so converting back to desired format
            file_name = os.path.join(self.BASE_DIR, save)
            #temp_image = os.path.join(self.BASE_DIR, "temp_image_html.html")
            if save.rsplit(".")[-1] != "html":
                raise  ValueError("Currently only HTML is supported")
            net.save(file_name)
            console.print(f"Image succesfully saved at {file_name}")
            #if html is False and os.path.isfile(temp_image): os.remove(temp_image)
        return net.display()


    def LigPlot2D(self,protein=None, ligand=None, verbose=False, save=False, html=False):
        if protein is None and self.protein is not None:
            protein = self.protein
        prot = self.__prep_protein(protein)
        if ligand is None and self.ligand is not None:
            ligand = self.ligand
        lig_suppl = self.__prep_ligand(ligand)
        df = self.fingerprint(prot, lig_suppl, verbose=verbose)
        try:
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
        except AttributeError:
            pass
        return self.__view_2D(df, lig_suppl[0], save=save, html=html)



if __name__ == "__main__":
    print("Use as module not as script")