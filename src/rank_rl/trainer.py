import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Union
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.trainer_utils import PredictionOutput
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    HfArgumentParser,
    EarlyStoppingCallback,
    GenerationConfig
)
from trl.models import create_reference_model
from tqdm import tqdm
from accelerate.utils import broadcast_object_list
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
from rdkit import Chem
from selfies import decoder
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
    """

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    return fp


def tanimoto_scores(
    true_selfies: str,
    pred_selfies: str
) -> list:
    try:
        true_smiles = decoder(true_selfies)
        true_mol = Chem.MolFromSmiles(true_smiles)
        if true_mol is None:
            return 0
    except Exception as e:
        return 0
    
    try:
        pred_smiles = decoder(pred_selfies)
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        if pred_mol is None:
            return 0
    except Exception as e:
        return 0
    
    true_fp = morgan_fp(true_mol)
    pred_fp = morgan_fp(pred_mol)

    try:
        reward = TanimotoSimilarity(true_fp, pred_fp)
    except Exception as e:
        print(f"Tanimoto error: {e}")
        reward = 0
    return reward


# Refer:  https://github.com/zjunlp/MolGen/blob/main/MolGen/finetune.py

def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=True, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            # loss(x1,x2,y)=max(0,−y*(x1−x2)+margin)
            # When y = 1 , we require that x1 > x2 + margin (i.e., the score of positive samples should be at least margin higher than that of negative samples).
            # if x1 - x2 > margin, the loss is 0, otherwise, the loss is margin - (x1 - x2)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss

class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss

class RankLossSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_generation = self.args.num_generation
        self.temperature = self.args.temperature
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True, temperature=self.temperature, 
            num_return_sequences=self.num_generation,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id
        )
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.args.smooth, ignore_index=-100)
        if self.args.loss_type == "kl":
            self.ref_model = create_reference_model(self.model)
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get inputs
        input_ids = inputs.get("input_ids") # [bz, seq_len]
        attention_mask = inputs.get("attention_mask") # [bz, seq_len]
        bz = input_ids.size(0)
        labels = inputs.get("labels") # [bz, seq_len]
        pad_token_id = self.processing_class.pad_token_id
        num_generation = self.args.num_generation
        labels_with_pad_token = torch.where(labels != -100, labels, pad_token_id)
        decoder_input_ids = inputs.get("decoder_input_ids") # [bz, seq_len]

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(model)
            candidates_ids = unwrapped_model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config
            ) # candidates: [bz x num_generation, seq_len]
            candidates_text = [self.processing_class.decode(x, skip_special_tokens=True).replace(" ", "") for x in candidates_ids]
            labels_text = [self.processing_class.decode(x, skip_special_tokens=True).replace(" ", "") for x in labels_with_pad_token]
        sorted_candidates_ids = []
        for i in range(bz):
            text_and_ids = [
                [candidates_text[i * num_generation + j], candidates_ids[i * num_generation + j]]
                for j in range(num_generation)
            ]
            gold_selfies = labels_text[i]
            sorted_text_and_ids = sorted(text_and_ids, key=lambda x: tanimoto_scores(gold_selfies, x[0]), reverse=True)
            sorted_ids = [x[1] for x in sorted_text_and_ids]
            sorted_candidates_ids.extend(sorted_ids)
        sorted_candidates_ids = torch.stack(sorted_candidates_ids, dim=0)

        cand_mask = sorted_candidates_ids != pad_token_id
        cand_mask[:, 0] = 1
        # Get model output
        input_ids_expanded = input_ids.unsqueeze(1).repeat(1, num_generation, 1)  # Repeat num_generation times along dimension 1
        input_ids_expanded = input_ids_expanded.contiguous().view(-1, input_ids_expanded.size(2))  # [bz x num_generation, seq_len]
        attention_mask_expanded = attention_mask.unsqueeze(1).repeat(1, num_generation, 1)  # Repeat num_generation times along dimension 1
        attention_mask_expanded = attention_mask_expanded.contiguous().view(-1, attention_mask_expanded.size(2))

        seq2seq_output = model(
            input_ids=input_ids_expanded,
            attention_mask=attention_mask_expanded,
            decoder_input_ids=sorted_candidates_ids,
            decoder_attention_mask=cand_mask,
        )
        model_logits = seq2seq_output.logits # [bz x cand_num, seq_len, word_dim]
        logits = model_logits.view(-1, num_generation, model_logits.size(1), model_logits.size(2)) # [bz, cand_num, seq_len, word_dim]
        logits = logits[:, :, :-1, :]  # truncate last token (next token)

        candidate_id = sorted_candidates_ids.contiguous().view(-1, num_generation, sorted_candidates_ids.size(1)) # [bz, cand_num, seq_len]
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.processing_class.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1) # [bz, cand_num, seq_len, 1]
        
        if self.args.normalize:
            if self.args.score_mode == "log":
                _logits = F.log_softmax(logits, dim=3)
            else:
                _logits = F.softmax(logits, dim=3)
            # torch.gather(input, dim, index) selects values from input tensor based on the indices provided in index tensor along a specified dimension dim
            scores = torch.gather(_logits, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(logits, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1)) ** self.args.length_penalty) # [bz, cand_num]

        # rank loss 
        # default cand_margin:0.001 gold_margin: 0 gold_weight: 0
        ranking_loss = RankingLoss(scores, None, self.args.cand_margin, self.args.gold_margin, self.args.gold_weight, True)
    
        # src_id_no_repeat = src_id[::num_generation]  # Take every num_generation-th element
        # gold_ids[gold_ids == self.processing_class.pad_token_id] = -100
        # sequence_output = model(
        #     input_ids=src_id_no_repeat,
        #     labels=gold_ids,
        # )
        # gold_logits = sequence_output.logits
        # mle_loss = self.loss_fct(gold_logits.view(-1, gold_logits.size(-1)), gold_ids.view(-1))
        if self.args.loss_type == "mle":
            mle_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            logits = mle_output.logits.contiguous()
            mle_loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = self.args.rank_weight * ranking_loss + self.args.mle_weight * mle_loss
        elif self.args.loss_type == "kl":
            model_log_probs = F.log_softmax(model_logits / self.temperature, dim=-1) # [bz x cand_num, seq_len, word_dim]
            with torch.no_grad():
                ref_cand_mask = sorted_candidates_ids != pad_token_id
                ref_cand_mask[:, 0] = 1
                ref_outputs = self.ref_model(
                    input_ids=input_ids_expanded,
                    attention_mask=attention_mask_expanded,
                    decoder_input_ids=sorted_candidates_ids,
                    decoder_attention_mask=ref_cand_mask,
                )
                ref_probs = F.softmax(ref_outputs.logits / self.temperature, dim=-1) # [bz x cand_num, seq_len, word_dim]
                kl_loss = nn.KLDivLoss(reduction="batchmean")(
                    model_log_probs, ref_probs
                ) * (self.temperature ** 2)
            loss = self.args.rank_weight * ranking_loss + self.args.kl_weight * kl_loss
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # greedy decoding
        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id
        )

        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        is_main_process = self.accelerator.is_main_process
        generated_outputs = []
        labels = []

        for batch in tqdm(eval_dataloader, desc=f"Evaluating: ", disable=not is_main_process):
            batch = self._prepare_inputs(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            pred_texts = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = self.processing_class.batch_decode(
                torch.where(batch["labels"] != -100, batch["labels"], self.processing_class.pad_token_id),
                skip_special_tokens=True
            )

            # Gather before leaving the batch so Accelerate removes distributed padding.
            pred_texts = self.accelerator.gather_for_metrics(pred_texts)
            label_texts = self.accelerator.gather_for_metrics(label_texts)
            generated_outputs.extend([t.replace(" ", "") for t in pred_texts])
            labels.extend([t.replace(" ", "") for t in label_texts])

        data_to_broadcast = [None]
        if is_main_process:
            metrics = self.compute_metrics((generated_outputs, labels))
            metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
            data_to_broadcast[0] = metrics
        
        broadcast_object_list(data_to_broadcast, from_process=0)
        metrics = data_to_broadcast[0]
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
