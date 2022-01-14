import numpy as np
from urllib.request import urlopen
from PIL import Image
import pandas as pd
from typing import Union, Optional
import io
import os
import subprocess


# PUBCHEM RELATED


class Attributes:
    def __init__(self, CID, format="csv"):

        self.CID = CID
        self.format = format
        self.CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

    @property
    def image(self):
        IMAGE_API = f"{self.CID_URL}/{self.CID}/record/png"
        self.image = Image.open(IMAGE_API)
        return self.image

    @property
    def description(self):
        DESC_API = f"{self.CID_URL}/{self.CID}/description/XML"
        return pd.read_xml(urlopen(DESC_API).read().decode("utf-8"))

    @property
    def formula(self):
        FORMULA_API = (
            f"{self.CID_URL}/{self.CID}/property/MolecularFormula/{self.format}"
        )
        return pd.read_csv(io.StringIO(urlopen(FORMULA_API).read().decode("utf-8")))

    @property
    def weight(self):
        MOL_WEIGHT_API = (
            f"{self.CID_URL}/{self.CID}/property/MolecularWeight/{self.format}"
        )
        return pd.read_csv(io.StringIO(urlopen(MOL_WEIGHT_API).read().decode("utf-8")))

    @property
    def xlog(self):
        XLOG_API = f"{self.CID_URL}/{self.CID}/property/XLogP/{self.format}"
        return pd.read_csv(io.StringIO(urlopen(XLOG_API).read().decode("utf-8")))

    @property
    def smile(self):
        ISO_SMILES_API = (
            f"{self.CID_URL}/{self.CID}/property/IsomericSmiles/{self.format}"
        )
        return pd.read_csv(io.StringIO(urlopen(ISO_SMILES_API).read().decode("utf-8")))

    def structure(self, save=False, *args, **kw):
        SDF_API = f"{self.CID_URL}/{self.CID}/SDF"
        self.structure = urlopen(SDF_API).read().decode("utf-8")

        if save:
            try:
                _dir = kw.get("dir", None)
                _filename = kw.get("filename", None)
                _smile_save = kw.get("smile_save", False)
                _dir = f"{_dir}" if _dir else "./data/structures"
                _filename = (
                    f"{_filename}"
                    if _filename
                    else "{}".format(self.structure.partition("\n")[0].strip())
                )

                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                FileExists = f"{_dir}/{_filename}.sdf"
                if _smile_save:
                    with open("./data/structures/smiles.smi", "a+") as f:
                        print(self.smile, file=f)
                with open(f"{FileExists}", "w") as w:
                    w.write(self.structure)
                    w.close()
                    return f"save success at {_dir} as {_filename}.sdf"
            except Exception as error:
                print(error)
        return self.structure

    # //TODO Method Chaining

    # def protein(self):
    #   return  f"https://pubchem.ncbi.nlm.nih.gov/protein/{self.CID}"
    # def gene(self):
    #   return  f"https://pubchem.ncbi.nlm.nih.gov/gene/{self.CID}"


def download_CID_structures(UNIQUE_ID):
    error_list = []
    print(f"\r=> Calling API and Downloading Structures...")
    for label, content in enumerate(UNIQUE_ID):
        try:
            t = Attributes(content)
            t.structure(dir="./data/structures", smile_save=True, save=True)
            if not os.path.exists(f"./data/structures/{content}.sdf"):
                error_list.append(content)
        except Exception as error:
            print(error)
            error_list.append(content)
        print(
            f"\r=> => Saved Successfully Verified: {label- len(error_list)} OK 👌 Error:"
            f" {len(error_list)}",
            end="",
            flush=True,
        )
    return error_list


def concatenate_aid_details(AID: [pd.DataFrame, list], download: bool = False):

    print("Time depends on number and size of files...Please be patient... ")
    total = len(AID)
    aid_list = []
    error_aid_list = []
    DATA = AID["aid"] if isinstance(AID, pd.DataFrame) else AID
    for count, _aid in enumerate(DATA):
        try:
            _aid_exp_detail = extract_aid_detail(f"{_aid}", download=download)
            aid_list.append(_aid_exp_detail)
        except Exception as error:
            # print(f"\r🔴{error}", sep=' ', end='', flush=True)
            error_aid_list.append(_aid)
            continue
        print(
            f"\rSuccess: {len(aid_list)}/{total} OK 👌 Error:"
            f" {len(error_aid_list)}/{total} 🔴             "
            f" {'Completed' if {count} != {total} else ''}",
            sep=" ",
            end="",
            flush=True,
        )

    print(f"Parsed : {len(aid_list) + len(error_aid_list)} 🚦 ")

    detail_data_type = {
        "AID": int,
        "Panel Member ID": int,
        "SID": int,
        "CID": int,
        "Bioactivity Outcome": str,
        "Target GI": int,
        "Target GeneID": int,
        "Activity Value [uM]": float,
        "Activity Name": str,
        "Assay Name": str,
        "Bioassay Type": str,
        "PubMed ID": str,
        "RNAi": str,
    }

    main_df = pd.DataFrame()
    empty_error = []
    for file in aid_list:
        try:
            main_df = main_df.append(
                pd.read_csv(file, dtype=detail_data_type, engine="python")
            )
            # print(main_df)
        except:
            empty_error.append(file)
            continue

    # main_df = pd.concat(
    #    [pd.read_csv(file, dtype=detail_data_type, engine="python", quoting=3, error_bad_lines=False) for file in aid_list if pd.read_csv(file).empty == False], ignore_index=True, sort=False)

    return main_df, error_aid_list


def extract_aid_detail(AID: int, download: bool = False, view: bool = False) -> str:

    _API = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/concise/CSV"
    if download:
        BASE_DIR = os.getcwd()
        path = "src/Data/Data_Source/PubChem"
        download_path = os.path.join(BASE_DIR, path)
        file = f"{AID}.csv"
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        if not os.path.exists(f"{download_path}/{file}"):
            command = f"wget -q {_API} -O {download_path}/{file}"
            subprocess.run(command, shell=True)
        downloaded_aid = f"{download_path}/{file}"
        print(f"\rDownloaded AID:{AID}.csv  ", sep=" ", end="", flush=True)
    if not view:
        return downloaded_aid
    f = urlopen(_API)
    return (f.read().decode("utf-8")), downloaded_aid
    # except Exception as error:
    #    print(f"\t 🔴{AID}_Error: {error}", sep=' ', end='', flush=True)
    #    raise ValueError('A very specific bad thing happened with request.')
    #    return -1


def check_non_downloaded(AID: pd.DataFrame) -> list:
    """AID is the main list of ids to be downloaded"""

    total = len(AID)
    downloaded_list = []
    error_aid_list = []
    BASE_DIR = os.getcwd()
    for count, _aid in enumerate(AID["aid"]):
        if os.path.exists(f"{BASE_DIR}/Data/PubChem/{_aid}.csv"):
            downloaded_list.append(_aid)
        else:
            error_aid_list.append(_aid)
        print(
            f"\rSuccessfully Downloaded : {len(downloaded_list)}/{total} OK 👌 Error:"
            f" {len(error_aid_list)}/{total} 🔴 ",
            sep=" ",
            end="",
            flush=True,
        )
    return error_aid_list


def main(input_file: str) -> pd.DataFrame:
    print("started.....")
    df = pd.read_csv(input_file, low_memory=False)
    download_list, error_list = concatenate_aid_details(df, download=True)


if __name__ == "__main":
    import sys

    print("started...")
    main(sys.argv[1])
