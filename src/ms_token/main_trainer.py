#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Refer to: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
import matchms
import matchms.filtering as ms_filters
from datasets import load_dataset, DatasetDict, load_from_disk
from collections import defaultdict
import torch
from copy import deepcopy

import transformers
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    HfArgumentParser,
    EarlyStoppingCallback,
    GenerationConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry

sys.path.append(".")
from src.metric import MoleculeEvaluator
from src.utils import save_arr

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: str = field(
        metadata={"help": "Path to pretrained model tokenizer"}
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    debug_flag: bool = field(default=False, metadata={"help": "Debug mode."})


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    early_stopping_patience: int = field(default=3, metadata={"help": "Step patience for early stopping"})


class MoleculeSeq2SeqTrainer(Seq2SeqTrainer):
    def evaluation_loop(self, *args, **kwargs):
        # One molecule is generated per input, so remove distributed padding.
        self.gather_function = self.accelerator.gather_for_metrics
        return super().evaluation_loop(*args, **kwargs)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    set_seed(training_args.seed)
    if training_args.should_log:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Data parameters {data_args}")

    
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets = load_dataset("csv",data_files=data_files,delimiter="\t")
    tokenizer_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        vocab_size=len(tokenizer), # !!!!! Very important !!!!! RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    training_args.generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.4,
        num_return_sequences=1,
        num_beams=1,
        early_stopping=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
    if training_args.should_log:
        print("=="*24, "Model Config", "=="*24)
        logger.info(f"{model.config}")
        logger.info(f"Generation config: {training_args.generation_config}")
        logger.info(f"Tokenizer is fast: {tokenizer.is_fast}")
        logger.info(f"Model vocab size matches tokenizer: {model.config.vocab_size == len(tokenizer)}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")


    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples["fps"])):
            if examples["fps"][i] and examples["selfies"][i]:
                inputs.append(examples["fps"][i])
                targets.append(examples["selfies"][i])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="longest", truncation=True)
        labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding="longest", truncation=True)

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    #  https://github.com/huggingface/transformers/blob/800510c67bfc5cedd0bb7635648a07f39719be43/examples/pytorch/summarization/run_summarization.py#L581
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns= train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns= eval_dataset.column_names, 
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns= predict_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    
    if training_args.should_log: 
        logger.info("Training Examples")
        for index in range(3):
            sample = train_dataset[index]
            input_ids = [id if id != -100 else tokenizer.pad_token_id for id in sample["input_ids"]]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            label_ids = [id if id != -100 else tokenizer.pad_token_id for id in sample["labels"]]
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            logger.info(f"Input text: {input_text}")
            logger.info(f"Labels: {label_text}")
            logger.info("="*50)

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    def postprocess_text(preds, labels):
        preds = [pred.replace(" ", "") for pred in preds]
        labels = [label.replace(" ", "") for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        evaluator = MoleculeEvaluator()
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = evaluator.evaluate_de_novo_step_selfies(decoded_preds, decoded_labels)

        if not training_args.do_train:
            final_results = []
            for i in range(len(decoded_labels)):
                final_results.append({
                    "pred": decoded_preds[i],
                    "refer": decoded_labels[i]
                })
            folder_path = "logs/results/{}".format(training_args.run_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_arr(final_results, os.path.join(folder_path, "{}.jsonl".format(training_args.run_name)))
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result
    # Initialize our Trainer
    trainer = MoleculeSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[
            # If there is no improvement in 3 evaluation steps, trigger early stopping. Ensure to set load_best_model_at_end=True, and it is recommended that save_step and eval_step be consistent.
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)
        ],
    )

    # Training
    if training_args.do_train:
        last_checkpoint = None
        # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if training_args.resume_from_checkpoint is not None:
            last_checkpoint = training_args.resume_from_checkpoint
        else:
            if os.path.isdir(training_args.output_dir):
                import shutil
                try:
                    shutil.rmtree(training_args.output_dir)
                    print(f"Folder {training_args.output_dir} has been deleted successfully.")
                except Exception as e:
                    print(f"An error occurred while deleting the folder: {e}")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip().replace(" ", "") for pred in predictions]

                 # Get and process labels
                labels = predict_results.label_ids
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip().replace(" ", "") for label in labels]

                batch_size = len(labels)
                if len(predictions) > batch_size:  # Top-K generation
                    num_return_sequences = len(predictions) // batch_size
                else:
                    num_return_sequences = 1

                save_jsonl = []
                for i in range(batch_size):
                    if num_return_sequences > 1:
                        pred = predictions[i * num_return_sequences : (i + 1) * num_return_sequences]
                    else:
                        pred = [predictions[i]]
                    save_jsonl.append({"pred": pred, "label": labels[i]})

                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.jsonl")
                save_arr(save_jsonl, output_prediction_file)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "MS Token Fine-tuning" }
    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
