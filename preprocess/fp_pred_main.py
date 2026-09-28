import argparse
import pandas as pd
from pathlib import Path
import copy
import numpy as np
import torch
from tqdm import tqdm
import selfies as sf
from rdkit import Chem

import sys
sys.path.append(".")
from mist.utils.plot_utils import *
import mist.subformulae.assign_subformulae as assign_subformulae
import mist.models.base as base
import mist.data.datasets as datasets
import mist.data.featurizers as featurizers

class MISTPredictor:
    def __init__(self, fp_ckpt, res_dir, mgf_input, labels,
                 weight_type="checkpoint", device="cuda:0", batch_size=None,
                 num_workers=32):
        self.fp_ckpt = fp_ckpt
        self.res_dir = Path(res_dir)
        self.mgf_input = mgf_input
        self.labels = labels
        self.res_dir.mkdir(exist_ok=True, parents=True)
        self.subform_dir = self.res_dir / "subforms_fp"
        self.subform_dir.mkdir(exist_ok=True, parents=True)

        self.weight_type = weight_type
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_dataset = None

        self.load_model()

    def assign_subformulae(self):
        assign_subformulae.assign_subforms(
            spec_files=self.mgf_input,
            labels_file=self.labels,
            output_dir=self.subform_dir,
            mass_diff_thresh=20,
            max_formulae=50,
            num_workers=self.num_workers,
            feature_id="FEATURE_ID",
            debug=False
        )

    def load_model(self):
        is_checkpoint = self.weight_type == "checkpoint"
        # Full Lightning checkpoints may contain pickled hyperparameters;
        # use checkpoint mode only for trusted files.
        fp_model = torch.load(
            self.fp_ckpt, map_location="cpu", weights_only=not is_checkpoint
        )
        if is_checkpoint:
            self.kwargs = copy.deepcopy(fp_model["hyper_parameters"])
            state_dict = fp_model["state_dict"]
        else:
            hidden_size, magma_modulo = {
                "mist-msg": (640, 2048),
                "mist-canopus": (512, 512),
            }[self.weight_type]
            # FRIGID configs/spec2mol_benchmark_{msg,canopus}.yaml:
            # https://github.com/coleygroup/FRIGID/tree/main/configs
            self.kwargs = dict(
                model="MistNet", fp_names=["morgan4096"],
                hidden_size=hidden_size, magma_modulo=magma_modulo,
                num_heads=8, peak_attn_layers=2, spectra_dropout=0.1,
                form_embedder="pos-cos", set_pooling="cls",
                iterative_preds="growing", refine_layers=4, top_layers=1,
                pairwise_featurization=True, embed_instrument=False,
                no_diffs=False, inten_transform="float", cls_type="ms1",
                max_peaks=None, shuffle_train=False,
            )
            state_dict = fp_model
        self.kwargs['device'] = str(self.device)
        self.kwargs['num_workers'] = 0
        self.kwargs['subform_folder'] = self.subform_dir
        self.kwargs['labels_file'] = self.labels
        if self.batch_size is not None:
            self.kwargs['batch_size'] = self.batch_size

        self.model = base.build_model(**self.kwargs)
        if not is_checkpoint:
            # FRIGID omits the training-only fingerprint permutations.
            state_dict['rand_ordering'] = torch.arange(self.model.output_size)
            state_dict['inv_ordering'] = torch.arange(self.model.output_size)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()


    def prepare_dataset(self):
        self.kwargs["spec_features"] = self.model.spec_features(mode="test") # 'peakformula_test'
        self.kwargs['mol_features'] = "none"
        self.kwargs['allow_none_smiles'] = True
        paired_featurizer = featurizers.get_paired_featurizer(**self.kwargs)

        spectra_mol_pairs = datasets.get_paired_spectra(**self.kwargs)
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))

        self.test_dataset = datasets.SpectraMolDataset(
            spectra_mol_list=spectra_mol_pairs, featurizer=paired_featurizer, **self.kwargs
        )

    def predict(self):
        self.prepare_dataset()
        output_preds = (
            self.model.encode_all_spectras(self.test_dataset, no_grad=True, **self.kwargs).cpu().numpy()
        )
        output_names = self.test_dataset.get_spectra_names()
        return output_preds, output_names

def parse_args():
    parser = argparse.ArgumentParser(description="Predict MIST fingerprints and encode SELFIES.")
    parser.add_argument("--weight-type", choices=["checkpoint", "mist-msg", "mist-canopus"],
                        default="checkpoint", help="Full trusted Lightning checkpoint or FRIGID weights")
    parser.add_argument("--fp-ckpt", type=Path, help="Override the selected weight file")
    parser.add_argument("--dataset-name", choices=["MassSpecGym", "CANOPUS"], default="MassSpecGym")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, help="Override inference batch size")
    parser.add_argument("--num-workers", type=int, default=32, help="Subformula assignment workers")
    parser.add_argument("--res-dir", type=Path, help="Directory for subformula files")
    parser.add_argument("--mgf-input", type=Path)
    parser.add_argument("--labels", type=Path, help="MIST spectrum labels TSV")
    parser.add_argument("--source-tsv", type=Path, help="MassSpecGym TSV or CANOPUS molecule labels TSV")
    parser.add_argument("--split-tsv", type=Path,
                        default=Path("data/datasets/CANOPUS/splits/canopus_hplus_100_0.tsv"))
    parser.add_argument("--output-dir", type=Path, help="Directory for fingerprint/SELFIES TSVs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name
    data_dir = Path("data") / dataset_name
    fp_ckpt = args.fp_ckpt or {
        "checkpoint": data_dir / "mist/mist.ckpt",
        "mist-msg": Path("data/frigid_pretrained_checkpoints/mist_msg.pt"),
        "mist-canopus": Path("data/frigid_pretrained_checkpoints/mist_canopus.pt"),
    }[args.weight_type]
    res_dir = args.res_dir or data_dir / "mist"
    mgf_input = args.mgf_input or data_dir / f"{dataset_name}.mgf"
    labels = args.labels or data_dir / f"{dataset_name}_labels.tsv"
    output_dir = args.output_dir or data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = MISTPredictor(
        fp_ckpt, res_dir, mgf_input, labels, weight_type=args.weight_type,
        device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    predictor.assign_subformulae()
    output_preds, output_names = predictor.predict()
    print(output_preds.shape, len(output_names))

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        indices_list = [np.where(row > threshold)[0].tolist() for row in output_preds]
        name_fps_keys = {name: fps for name, fps in zip(output_names, indices_list)}

        if dataset_name == "MassSpecGym":
            df = pd.read_csv(args.source_tsv or "data/MassSpecGym/data/MassSpecGym.tsv", sep='\t')
        elif dataset_name == "CANOPUS":
            canopus_split = pd.read_csv(args.split_tsv, sep='\t')
            canopus_labels = pd.read_csv(args.source_tsv or './data/datasets/CANOPUS/labels.tsv', sep='\t')
            canopus_labels["name"] = canopus_labels["spec"]
            df = canopus_labels.merge(canopus_split, on="name")
        else:
            raise ValueError("Unknown dataset name")
        print(f"Before processing, the number of entries is: {len(df)}")
        # Add FPS column/Selfies column
        df['fps'] = ''
        df['selfies'] = ''
        
        # Record the indices of rows that need to be deleted (those where FPS or Selfies generation failed)
        to_drop = []
        for idx, row in df.iterrows():
            identifier_key = "identifier" if dataset_name == "MassSpecGym" else "name"
            identifier = row[identifier_key]
            smiles = row['smiles']
            if identifier not in name_fps_keys:
                to_drop.append(idx)
                print(f"identifier {identifier} not in name_fps_keys")
                continue
            fps = name_fps_keys[identifier]
            df.at[idx, 'fps'] = "".join([f"<fp{fp:04d}>" for fp in fps])
            try:
                # Convert to standard SELFIES
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                selfies_str = sf.encoder(canonical_smiles)
                df.at[idx, 'selfies'] = selfies_str
            except:
                print(f"Error in encode smiles to selfies: {smiles}")
                to_drop.append(idx)
                continue
        # Delete rows where SELFIES generation failed
        df = df.drop(index=to_drop)
        print(f"Final number of entries: {len(df)}")

        df.to_csv(output_dir / f"{dataset_name}_fps_selfies_threshold_{threshold}.tsv", sep='\t', index=False)
