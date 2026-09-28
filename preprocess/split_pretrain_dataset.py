from datasets import load_dataset, DatasetDict
from collections import defaultdict
import pandas as pd
from datasets import DatasetDict
from pathlib import Path

def split_pubchem_data():
    # Configuration
    file_path = "logs/datasets/MassSpecGym/MassSpecGym_fps_selfies_pretrain_4M.tsv"
    validation_size = 10000  # Number of validation samples
    
    df = pd.read_csv(file_path, sep='\t')
    df = df.sample(frac=1, random_state=42)
    
    validation_set = df.iloc[:validation_size]
    train_set = df.iloc[validation_size:]
    
    validation_file = f"logs/datasets/MassSpecGym/pubchem-4M/val.tsv"
    train_file = f"logs/datasets/MassSpecGym/pubchem-4M/train.tsv"
    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    
    validation_set.to_csv(validation_file, sep='\t', index=False)
    train_set.to_csv(train_file, sep='\t', index=False)
    
    print(f"- Train: {train_file} ({len(train_set)} )") # 117655119
    print(f"- Val: {validation_file} ({len(validation_set)} )") # 10000

    # 4M 3912277 / 10000

if __name__ == "__main__":
    split_pubchem_data()
    
