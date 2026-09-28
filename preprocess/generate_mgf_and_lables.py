""" parse_utils.py """
from pathlib import Path
from typing import Tuple, List, Optional
from itertools import groupby

from tqdm import tqdm
import numpy as np
import pandas as pd


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


def parse_spectra_mgf(
    mgf_file: str, max_num: Optional[int] = None
) -> List[Tuple[dict, List[Tuple[str, np.ndarray]]]]:
    """parse_spectr_mgf.

    Parses spectra in the MGF file formate, with

    Args:
        mgf_file (str) : str
        max_num (Optional[int]): If set, only parse this many
    Return:
        List[Tuple[dict, List[Tuple[str, np.ndarray]]]]: metadata and list of spectra
            tuples containing name and array
    """

    key = lambda x: x.strip() == "BEGIN IONS"
    parsed_spectra = []
    with open(mgf_file, "r") as fp:

        for (is_header, group) in tqdm(groupby(fp, key)):

            if is_header:
                continue

            meta = dict()
            spectra = []
            # Note: Sometimes we have multiple scans
            # This mgf has them collapsed
            cur_spectra_name = "spec"
            cur_spectra = []
            group = list(group)
            for line in group:
                line = line.strip()
                if not line:
                    pass
                elif line == "END IONS" or line == "BEGIN IONS":
                    pass
                elif "=" in line:
                    k, v = [i.strip() for i in line.split("=", 1)]
                    meta[k] = v
                else:
                    mz, intens = line.split()
                    cur_spectra.append((float(mz), float(intens)))

            if len(cur_spectra) > 0:
                cur_spectra = np.vstack(cur_spectra)
                spectra.append((cur_spectra_name, cur_spectra))
                parsed_spectra.append((meta, spectra))
            else:
                pass
                # print("no spectra found for group: ", "".join(group))

            if max_num is not None and len(parsed_spectra) > max_num:
                # print("Breaking")
                break
        return parsed_spectra

if __name__ == "__main__":
    df = pd.read_csv(f"data/MassSpecGym/data/MassSpecGym.tsv", sep="\t")
    only_MH = False
    if only_MH:
        save_dir = "MassSpecGym"
    else:
        save_dir = "MassSpecGym-ALL"
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
    if only_MH:
        ION_LST = ["[M+H]+"]
    # https://github.com/samgoldman97/mist/blob/126649324655f00f8a8707f75a0c0492b11cbb4f/src/mist/utils/chem_utils.py#L79
    for _, row in df.iterrows():
        # ==================== Construct MGF metadata ====================
        adduct = row["adduct"]
        if adduct not in ION_LST: continue
        meta = {
            "FEATURE_ID": row["identifier"],
            "adduct": row["adduct"],
            "collision_energy": row["collision_energy"],
            # https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/viewer/main/val?sort%5Bcolumn%5D=identifier&sort%5Bdirection%5D=asc&sort%5Btransform%5D=length&p=64&row=6496
            "parentmass": float(row["precursor_mz"]),
        }
        
        # ==================== Parse spectral data ====================
        mzs = list(map(float, row["mzs"].split(",")))
        intensities = list(map(float, row["intensities"].split(",")))
        assert len(mzs) == len(intensities), "The number of m/z values does not match the number of intensities."
        peaks = np.column_stack((mzs, intensities))
        spec = [("ms2", peaks)]
        meta_spec_list.append((meta, spec))
        
        # ==================== Create label.tsv entries ====================
        label_entry = {
            "spec": str(row["identifier"]),        
            "formula": row["formula"],              
            "ionization": row["adduct"],           
            "dataset": "MassSpecGym",       
            "compound": f"{row['identifier']}", 
            "parentmass": float(row["precursor_mz"]),
            "instrument": row.get("instrument_type", "unknown")
        }
        label_entries.append(label_entry)
    
    # ==================== Generate MGF file content ====================
    mgf_output = build_mgf_str(meta_spec_list)
    Path(f"logs/datasets/{save_dir}").mkdir(parents=True, exist_ok=True)
    with open(f"logs/datasets/{save_dir}/{save_dir}.mgf", "w") as f:
        f.write(mgf_output)

    # ==================== Generate label.tsv ====================
    label_df = pd.DataFrame(label_entries)
    print("After processing, the number of entries is: ", len(label_entries))
    label_df = label_df[["spec", "formula", "ionization", "dataset", "compound", "parentmass", "instrument"]] 
    label_df.to_csv(f"logs/datasets/{save_dir}/{save_dir}_labels.tsv", sep="\t", index=False)
