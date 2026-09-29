import argparse
import os

import pandas as pd

def prepare_test_data_massspecgym(file_path, fold="test", dataset_name="MassSpecGym"):
    df = pd.read_csv(file_path, sep='\t')
    fold_key = "fold" if dataset_name == "MassSpecGym"  else "split"
    split_df = df[df[fold_key] == fold]
    if dataset_name == "MassSpecGym" and fold == "train":
        split_df = split_df[split_df["adduct"] == "[M+H]+"]
    
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    output_dir = os.path.join(file_dir, fold)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)

    split_df.to_csv(output_file, sep='\t', index=False)
    print(f"{fold}: {len(split_df)} rows -> {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split fingerprint/SELFIES tables by their original folds.")
    parser.add_argument("--dataset-name", choices=["MassSpecGym", "CANOPUS"], default="MassSpecGym")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Select an existing fingerprint table by threshold (default: 0.2)")
    parser.add_argument("--data-dir",
                        help="Directory containing the TSV (default: logs/datasets/<dataset-name>)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir or f"logs/datasets/{args.dataset_name}"
    file_path = os.path.join(data_dir, f"{args.dataset_name}_fps_selfies_threshold_{args.threshold}.tsv")
    for fold in ("train", "test", "val"):
        prepare_test_data_massspecgym(file_path, fold, args.dataset_name)
