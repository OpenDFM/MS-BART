from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import numpy as np
import selfies
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from pathlib import Path


RDLogger.DisableLog('rdApp.*')

# https://github.com/samgoldman97/mist/blob/126649324655f00f8a8707f75a0c0492b11cbb4f/src/mist/data/featurizers.py#L244
def get_morgan_4096(smile: str, nbits: int = 4096, radius=2):
    try:
        mol = AllChem.MolFromSmiles(smile)
    except Exception:
        return None
    if mol is None:
        return None
    
    def fp_fn(m):
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

    fingerprint = fp_fn(mol)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    ones_indices = np.where(array == 1)[0].tolist()
    fps_tokens = "".join([f"<fp{fp:04d}>" for fp in ones_indices])
    return fps_tokens

def get_selfies(smile: str):
    try:
        mol = Chem.MolFromSmiles(smile) 
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        selfies_str = selfies.encoder(canonical_smiles)
    except Exception:
        selfies_str = None
    return selfies_str

def process_single_smiles(smile):
    """Process a single SMILES string and return both fps and selfies"""
    fps = get_morgan_4096(smile)
    selfies = get_selfies(smile)
    if fps is None or selfies is None:
        print(f"Error processing SMILES: {smile}")
        return None
    return {'fps': fps, 'selfies': selfies}

def process_smiles_file(input_file, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file, sep='\t', engine='pyarrow')
    if 'smiles' not in df.columns:
        raise ValueError("Input file must contain a 'smiles' column")    
    # Prepare output data
    print("read files ok")
    # If the data volume is very large (such as 1.18 million SMILES), inter - process communication (IPC) will become a bottleneck.
    fps_list = []
    selfies_list = []
    
    cpu_count = 24
    chunk_size = 100000  # Process 100,000 entries each time.
    for i in tqdm(range(0, len(df), chunk_size), desc="chunk process"):
        chunk = df.iloc[i:i+chunk_size]
        with ProcessPool(max_workers=cpu_count) as pool:
            future = pool.map(process_single_smiles, chunk['smiles'])
            iterator = future.result()
            while True:
                try:
                    result = next(iterator)
                    if result is not None:
                        fps_list.append(result['fps'])
                        selfies_list.append(result['selfies'])
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(f"Function took longer than 120 seconds: {error}")
                except Exception as error:
                    print(f"Function raised {error}")

    # Create output DataFrame
    output_df = pd.DataFrame({
        'fps': fps_list,
        'selfies': selfies_list
    })
    output_df.to_csv(output_file, sep='\t', index=False)
    print(f"Successfully saved results to {output_file}")

if __name__ == '__main__':
    smile = 'CCO'
    print(get_morgan_4096(smile))
    print(get_selfies(smile))
    process_smiles_file(
        "data/MassSpecGym/data/molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv",
        "logs/datasets/MassSpecGym/MassSpecGym_fps_selfies_pretrain_4M.tsv"
    )
