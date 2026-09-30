from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from transformers import GenerationConfig
import torch
from tqdm import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
import json
import numpy as np
from rich.table import Table
from rich.console import Console
import os
from collections import Counter
from selfies import decoder
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys
sys.path.append(".")
from src.metric import MoleculeEvaluator, TopkMoleculeEvaluator
from src.utils import save_arr, selfies_to_formula
import re
from collections import defaultdict

def selfies_to_formula(selfies):
    try:
        smiles = decoder(selfies)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception as e:
        return None

def compare_formulas(formula1, formula2, ignore_h=False):
    def parse_formula(formula):
        # Parse formula, "C6H6" -> {'C':6, 'H':6}
        pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
        elements = pattern.findall(formula)
        counts = defaultdict(int)
        for elem, cnt in elements:
            if ignore_h and elem == 'H':
                continue
            counts[elem] += int(cnt) if cnt else 1
        return counts
    
    dict1 = parse_formula(formula1)
    dict2 = parse_formula(formula2)
    
    all_elements = set(dict1.keys()).union(set(dict2.keys()))
    differences = {}
    diff_cnt = 0
    for elem in all_elements:
        diff = abs(dict1.get(elem, 0) - dict2.get(elem, 0))
        if diff > 0:
            differences[elem] = diff
            diff_cnt += diff
    
    return differences, diff_cnt

if __name__ == "__main__":

    #Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate BART model for molecular generation")
    parser.add_argument("--model_path", type=str, default="logs/checkpoints/bart-base-4M-pretrain-ft", help="Path to the trained BART model")
    parser.add_argument("--test_path",  type=str, default="logs/datasets/MassSpecGym-ALL/test/Deduplicated_MassSpecGym-ALL_fps_selfies_threshold_0.2.tsv", help="Path to the test dataset")    
    parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for sampling")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument("--topk", type=int, default=10, help="Number of sequences to evaluate")
    parser.add_argument("--compute_mces", action="store_true", help="Compute mces molecular metrics")
    args = parser.parse_args()

    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()
    # Initialize evaluators only on main process
    if distributed_state.is_main_process:
        print("Evaluate with: ", args)

    model_path = args.model_path
    test_path = args.test_path
    temperature = args.temperature

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(distributed_state.device)
    model.eval()

    test_dataset = load_dataset("csv", data_files={"test": test_path }, delimiter="\t", keep_in_memory=True)["test"]
    debug = False
    if debug: test_dataset = test_dataset.select(range(1000))
    
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True, temperature=temperature, 
        num_return_sequences=args.num_beams,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    completions_per_process = []
    with distributed_state.split_between_processes(test_dataset, apply_padding=False) as batched_prompts:
        batch_size = 16
        if args.num_beams > 20 or args.num_beams > 20:
            batch_size = 4
        for i in tqdm(range(0, len(batched_prompts), batch_size), desc=f"Rank [{distributed_state.process_index}] processing: "):
            prompts_batch = batched_prompts[i:i + batch_size]
            fps = prompts_batch["fps"]
            selfies_true = prompts_batch["selfies"]
            formulas = prompts_batch["formula"]
            sample_ids_name = "identifier" if "MassSpecGym" in test_path else "name"
            sample_ids = prompts_batch[sample_ids_name]
            tip_inputs = tokenizer(fps, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
            tip_inputs = {k: v.to(distributed_state.device) for k, v in tip_inputs.items()}
            with torch.inference_mode():
                tip_completion_ids = model.generate(**tip_inputs, generation_config=generation_config)
            tip_completion_ids = tip_completion_ids.cpu()
            answers = [tokenizer.decode(x, skip_special_tokens=True).replace(" ", "") for x in tip_completion_ids]

            answers = [
                answers[j * args.num_beams:(j + 1) * args.num_beams]
                for j in range(len(fps))
            ]

            for st, sp, formula, sample_id in zip(selfies_true, answers, formulas, sample_ids):
                selfies_formulas = []
                for idx, s in enumerate(sp):
                    s_formula = selfies_to_formula(s)
                    if s_formula is not None:
                        _, diff_cnt = compare_formulas(formula, s_formula, ignore_h=False)
                        obj = {
                            "selfies": s,
                            "formula": s_formula,
                            "formula_diff": diff_cnt,
                            "index": idx,
                        }
                        selfies_formulas.append(obj)
                selfies_formulas_sorted = sorted(selfies_formulas, key=lambda x: (x['formula_diff'], x['index']))
                if len(selfies_formulas_sorted) == 0:
                    selfies_formulas_sorted = ["[C]"]
                else:
                    selfies_formulas_sorted = [s['selfies'] for s in selfies_formulas_sorted]

                completions_per_process.extend([
                    {
                        "selfies_true": st,
                        "selfies_pred": selfies_formulas_sorted,
                        "sample_id": sample_id
                    }
                ])

    torch.cuda.empty_cache()
    completions_gather = gather_object(completions_per_process)
    
    # Initialize evaluators only on main process
    if distributed_state.is_main_process:
        evaluator = MoleculeEvaluator(mces=args.compute_mces)
        topk_evaluator = TopkMoleculeEvaluator(mces=args.compute_mces)
        all_preds = [item["selfies_pred"] for item in completions_gather]
        all_labels = [item["selfies_true"] for item in completions_gather]
        sample_ids = [item["sample_id"] for item in completions_gather]

        # Save first
        model_name = os.path.basename(model_path)
        save_path = os.path.join("logs", "results", model_name + ".jsonl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save results
        save_predictions = []
        for i in range(len(all_labels)):
            save_predictions.append({
                "pred": all_preds[i],
                "label": all_labels[i],
                "sample_id": sample_ids[i]
            })
        save_arr(save_predictions, save_path)

        # Calculate Top1 metrics
        top1_selfies_pred = [s[0] for s in all_preds]
        top1_results = evaluator.evaluate_de_novo_step_selfies(top1_selfies_pred, all_labels)
        print("Top1 results:", top1_results)
        # generated token and gold token statistics
        top1_preds_token_lens = [len(tokenizer.tokenize(s)) for s in top1_selfies_pred]
        labels_token_lens = [len(tokenizer.tokenize(s)) for s in all_labels]

        console = Console()
        table = Table(title="Token Length Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Generated", style="magenta")
        table.add_column("Labels", style="green")
        table.add_row("Mean", f"{np.mean(top1_preds_token_lens)}", f"{np.mean(labels_token_lens)}")
        table.add_row("Max", f"{np.max(top1_preds_token_lens)}", f"{np.max(labels_token_lens)}")
        table.add_row("Min", f"{np.min(top1_preds_token_lens)}", f"{np.min(labels_token_lens)}")
        table.add_row("Std", f"{np.std(top1_preds_token_lens)}", f"{np.std(labels_token_lens)}")
        console.print(table)

        # Calculate Topk metrics
        k = args.topk
        topk_preds = [s[:k] if len(s) > k else s for s in all_preds]
        topk_results = topk_evaluator.evaluate_de_novo_step_selfies_top_k(topk_preds, all_labels)
        print("Topk results:", topk_results)

        metrics_path = os.path.splitext(save_path)[0] + ".metrics.json"
        with open(metrics_path, "w") as handle:
            json.dump({
                "config": vars(args),
                "num_samples": len(all_labels),
                "top1": top1_results,
                "topk": topk_results,
            }, handle, indent=2, allow_nan=False)
