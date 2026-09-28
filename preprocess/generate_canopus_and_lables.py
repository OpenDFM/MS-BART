""" parse_utils.py """
from pathlib import Path
from typing import Tuple, List, Optional
from itertools import groupby

from tqdm import tqdm
import numpy as np
import pandas as pd
import os

def build_mgf_str(
    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=["PEPMASS", "parentmass", "PRECURSOR_MZ"],
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec in tqdm(meta_spec_list):
        str_rows = ["BEGIN IONS"]

        # Try to add precusor mass
        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])
        else:
            raise NotImplementedError()
        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out


def parse_spectra(spectra_file: str) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
    """parse_spectra.

    Parses spectra in the SIRIUS format and returns

    Args:
        spectra_file (str): Name of spectra file to parse
    Return:
        Tuple[dict, List[Tuple[str, np.ndarray]]]: metadata and list of spectra
            tuples containing name and array
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                # Add new tuple
                spectras.append((spectra_header, peak_data))
        # Get meta data
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras



if __name__ == "__main__":

    # ==================== Merge the original data ====================
    # https://github.com/coleygroup/DiffMS/blob/6c6924f10cdbfa4badd10ed8fecede649c7ddcdc/data_processing/build_fp2mol_datasets.py#L73
    canopus_split = pd.read_csv('./data/canopus_train_export/splits/canopus_hplus_100_0.tsv', sep='\t')
    canopus_labels = pd.read_csv('./data/canopus_train_export/labels.tsv', sep='\t')
    canopus_labels["name"] = canopus_labels["spec"]
    df = canopus_labels.merge(canopus_split, on="name")

    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]] = []
    label_entries = []
    print("Before processing, the number of entries is: ", len(df))

    ION_LST = [
        "[M+H]+",
        "[M+Na]+",
        "[M+K]+",
        "[M-H2O+H]+",
        "[M+H3N+H]+",
        "[M]+",
        "[M-H4O2+H]+",
    ] 
    ION_LST = [ "[M+H]+" ]
    # https://github.com/samgoldman97/mist/blob/126649324655f00f8a8707f75a0c0492b11cbb4f/src/mist/utils/chem_utils.py#L79
    for _, row in df.iterrows():
        # ==================== Construct MGF metadata ====================
        adduct = row["ionization"]
        if adduct not in ION_LST: continue
        meta_data, spectras = parse_spectra(os.path.join("data/canopus_train_export/spec_files", row["name"]+".ms"))
        meta = {
            "FEATURE_ID": row["name"],
            "adduct": row["ionization"],
            "collision_energy": meta_data.get("collision_energy", 20),
            # https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/viewer/main/val?sort%5Bcolumn%5D=identifier&sort%5Bdirection%5D=asc&sort%5Btransform%5D=length&p=64&row=6496
            # https://github.com/samgoldman97/mist/blob/126649324655f00f8a8707f75a0c0492b11cbb4f/src/mist/utils/spectra_utils.py#L110
            "parentmass": meta_data.get("parentmass", 1000000),
        }
        
        # ==================== Parse spectral data ====================
        spec = [("ms2", spectras[0][1])]
        meta_spec_list.append((meta, spec))
        
        # ==================== Construct label.tsv entries ====================
        label_entry = {
            "spec": str(row["name"]),         
            "formula": row["formula"],            
            "ionization": row["ionization"],            
            "dataset": "CANOPUS",       
            "compound": f"{row['name']}",  
            "parentmass": meta_data.get("parentmass", 1000000), 
            "instrument": row.get("instrument", "unknown")
        }
        label_entries.append(label_entry)
    
    # ==================== Generate MGF file content ====================
    mgf_output = build_mgf_str(meta_spec_list)
    Path("logs/datasets/CANOPUS").mkdir(parents=True, exist_ok=True)
    with open(f"logs/datasets/CANOPUS/CANOPUS.mgf", "w") as f:
        f.write(mgf_output)

    # ==================== Generate label.tsv ====================
    label_df = pd.DataFrame(label_entries)
    print("After processing, the number of entries is: ", len(label_entries))
    label_df = label_df[["spec", "formula", "ionization", "dataset", "compound", "parentmass", "instrument"]]
    label_df.to_csv(f"logs/datasets/CANOPUS/CANOPUS_labels.tsv", sep="\t", index=False)
