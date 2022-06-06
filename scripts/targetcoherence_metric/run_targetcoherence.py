# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import copy
from dataclasses import dataclass, field
from typing import Optional
import csv
import numpy as np
from scipy.special import softmax
from datasets import load_dataset, load_metric, Dataset
# from results_analyses_tools import compute_metrics
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # EvalPrediction,
    # HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# from transformers.trainer_utils import is_main_process
# logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda')

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
transformers.logging.set_verbosity(transformers.logging.WARNING)
transformers.logging.get_verbosity() 

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}



columns_to_remove = ['id','adversarial_negative_responses', 'random_negative_responses', 'positive_responses', 'context']



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )

    type_negative: Optional[str] = field(
        default='random_negative_responses', metadata={"help": "A csv or a json file containing the test data."}
    )

    type_negative_extra: Optional[str] = field(
        default='', metadata={"help": "A csv or a json file containing the test data."}
    )

    type_negative_test: Optional[str] = field(
        default='random_negative_responses', metadata={"help": "A csv or a json file containing the test data."}
    )




    # def __post_init__(self):
    #     if self.task_name is not None:
    #         self.task_name = self.task_name.lower()
    #         if self.task_name not in task_to_keys.keys():
    #             raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
    #     elif  self.validation_file is None:
    #         raise ValueError("Need either a GLUE task or a training/validation file.")
    #     else:
    #         # extension = self.train_file.split(".")[-1]
    #         # assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         extension = self.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.output_dir == '':
        training_args.output_dir = model_args.model_name_or_path

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    # )

    # training_args.save_steps=2000
    # training_args.eval_steps=2000
    # training_args.evaluate_during_training=True
    # training_args.evaluation_strategy = 'epoch'
    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):
    #     # transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.train_file is not None:
        if data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset("glue", data_args.task_name)
        elif data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset(
                "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
            )
        else:
            # Loading a dataset from local json files
            # datasets = load_dataset(
            #     "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
            # )

            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            # if data_args.validation_file is not None:
            #     data_files["validation"] = data_args.validation_file
            extension = data_args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            datasets = load_dataset(extension, data_files=data_files)

        datasets = datasets.map(preprocess_function, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)

        train_dataset = datasets["train"]
        print(len(train_dataset), 'len train_dataset')
            # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        label_list = [0,1]
        num_labels = 2
        is_regression = False
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        # if is_regression:
        #     num_labels = 1
        # else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            # label_list = datasets["train"].unique("label")
            # label_list.sort()  # Let's sort it for determinism
            # num_labels = len(label_list)

            

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    special_tokens_dict = {'additional_special_tokens': ['<eot>']}
    # special_tokens_dict = {'additional_special_tokens': ['<speaker1>','<speaker2>']}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # # Preprocessing the datasets
    # if data_args.task_name is not None:
    #     sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    # else:
    #     # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    #     non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    #     if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
    #         sentence1_key, sentence2_key = "sentence1", "sentence2"
    #     else:
    #         if len(non_label_column_names) >= 2:
    #             sentence1_key, sentence2_key = non_label_column_names[:2]
    #         else:
    #             sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function_org(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    def get_string_text(tokens_a, tokens_b):
        max_num_tokens = data_args.max_seq_length - 3
        total_length = len(tokens_a) + len(tokens_b)
        if total_length > max_num_tokens:
            len_b = len(tokens_b)
            a_begin = max_num_tokens - len_b
            tokens_a = tokens_a[-a_begin:]

        try:
            assert len(tokens_a) + len(tokens_b) <= max_num_tokens
            assert len(tokens_a) >= 1
        except:
            import pdb;pdb.set_trace()
            print('some problem with preproc')
        #assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append(tokenizer.cls_token)
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append(tokenizer.sep_token)
        segment_ids.append(1)

        return tokens, segment_ids

    def preprocess_function(examples):
        # Tokenize the texts
        processed = []
        items = []
        keys = list(examples.keys())
        # print(examples['context'])
        for i in range(len(examples[keys[0]])):
            ex = {}
            for k in keys:
                ex[k] = examples[k][i]
            items.append(ex)
        # print(len(items))
        all_segment_ids = []
        all_texts = []
        nsp_labels = []
        # speaker1, speaker2 = "<speaker1>", "<speaker2>"
        speaker1, speaker2 = "<eot>", "<eot>"
        for example in items:
            context_arr = [line for line in example['context'] if len(line) > 0 and not line.isspace()]
            context_arr = [speaker2 + ' ' + s if (len(context_arr) - i+1) % 2 else speaker1 + ' ' + s for i, s in enumerate(context_arr)]
            context = ' '.join(context_arr).lower()

            positive_responses = example['positive_responses']

            if data_args.type_negative == 'adversarial_negative_responses':
                type_negative = 'adversarial_negative_responses'
            else:
                type_negative = 'random_negative_responses'
                #we will handle case of both random and adversarial below

            negative_responses = example[type_negative]
            for i in range(len(positive_responses)):
                positive_response = positive_responses[i].lower()
                negative_response = negative_responses[i].lower()
                tokens_a = context

                tokens_a = tokenizer.tokenize(tokens_a)

                tokens_b = positive_response
                tokens_b = tokenizer.tokenize(speaker2 + ' ' + tokens_b, max_length=data_args.max_seq_length-5)
                tokens, segment_ids = get_string_text(tokens_a, tokens_b)
                all_texts.append(tokens)
                all_segment_ids.append(segment_ids)
                nsp_labels.append(0)

                tokens_b = negative_response
                tokens_b = tokenizer.tokenize(speaker2 + ' ' +tokens_b, max_length=data_args.max_seq_length-5)
                tokens, segment_ids = get_string_text(tokens_a, tokens_b)
                all_texts.append(tokens)
                all_segment_ids.append(segment_ids)
                nsp_labels.append(1)

                if data_args.type_negative == 'both':
                    negative_response = example[data_args.type_negative_extra][i].lower()
                    tokens_b = negative_response
                    tokens_b = tokenizer.tokenize(speaker2 + ' ' +tokens_b, max_length=data_args.max_seq_length-5)
                    tokens, segment_ids = get_string_text(tokens_a, tokens_b)
                    all_texts.append(tokens)
                    all_segment_ids.append(segment_ids)
                    nsp_labels.append(1)

            #add extra negative responses if present
            if data_args.type_negative == 'both':
                for i in range(len(positive_responses), len(example[data_args.type_negative_extra])):
                    negative_response = example[data_args.type_negative_extra][i].lower()
                    tokens_b = negative_response
                    tokens_b = tokenizer.tokenize(speaker2 + ' ' +tokens_b, max_length=data_args.max_seq_length-5)
                    tokens, segment_ids = get_string_text(tokens_a, tokens_b)
                    all_texts.append(tokens)
                    all_segment_ids.append(segment_ids)
                    nsp_labels.append(1)

            # print(len(all_texts), print(range(len(positive_responses), len(example[data_args.type_negative_extra]))))

        tokenized = tokenizer.batch_encode_plus(
            all_texts,
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # token_type_ids=segment_ids,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask,
            is_split_into_words=True,
            return_special_tokens_mask=True,
            add_special_tokens=False,
        )

        # print(len(tokenized['input_ids']))
        padded_length = len(tokenized['input_ids'][0])
        all_segment_ids = [ x + [0] * (padded_length-len(x)) for x in all_segment_ids]
        tokenized['token_type_ids'] = all_segment_ids
        tokenized['label'] = nsp_labels
        # import pdb;pdb.set_trace()            
        # return processed

        to_remove_cols = ['context_string', 'adv_gen_neg_responses_t1', 'adv_gen_neg_responses_k1', 'positive_responses_seqgen', 'adv_gen_neg_responses_k1_seqgen', 'bm25_sampled', 'sentenceBERT_sampled', 'random_sampled']
        for colrname in to_remove_cols:
            if colrname in examples:
                del examples[colrname]


        return tokenized



    def preprocess_function_test(examples):
        # Tokenize the texts
        processed = []
        items = []
        keys = list(examples.keys())
        # print(examples['context'])
        for i in range(len(examples[keys[0]])):
            ex = {}
            for k in keys:
                ex[k] = examples[k][i]
            items.append(ex)
        # print(len(items))
        all_segment_ids = []
        all_texts = []
        nsp_labels = []
        context_texts = []
        response_texts = []

        speaker1, speaker2 = "<speaker1>", "<speaker2>"
        speaker1, speaker2 = "<eot>", "<eot>"

        for example in items:
            context_arr = [line for line in example['context'] if len(line) > 0 and not line.isspace()]
            context_arr = [speaker2 + ' ' + s if (len(context_arr) - i+1) % 2 else speaker1 + ' ' + s for i, s in enumerate(context_arr)]
            context = ' '.join(context_arr)

            positive_responses = example['positive_responses']
            # negative_responses = example['adversarial_negative_responses']
            # print('type ', data_args.type_negative_test)
            negative_responses = example[data_args.type_negative_test]

            for i in range(len(positive_responses)):
                positive_response = positive_responses[i]
                negative_response = negative_responses[i]
                tokens_a = context

                tokens_a = tokenizer.tokenize(tokens_a)

                tokens_b = positive_response
                tokens_b = tokenizer.tokenize(speaker2 + ' ' +tokens_b, max_length=data_args.max_seq_length-5)
                tokens, segment_ids = get_string_text(tokens_a, tokens_b)
                all_texts.append(tokens)
                all_segment_ids.append(segment_ids)
                nsp_labels.append(0)
                context_texts.append(' -- '.join(context_arr))
                response_texts.append(positive_response)


                tokens_b = negative_response
                tokens_b = tokenizer.tokenize(speaker2 + ' ' + tokens_b, max_length=data_args.max_seq_length-5)
                tokens, segment_ids = get_string_text(tokens_a, tokens_b)
                all_texts.append(tokens)
                # print(example, i)
                # print(context, tokens)
                all_segment_ids.append(segment_ids)
                nsp_labels.append(1)
                context_texts.append(' -- '.join(context_arr))
                response_texts.append(negative_response)


        # import pdb;pdb.set_trace()            
        tokenized = tokenizer.batch_encode_plus(
            all_texts,
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # token_type_ids=segment_ids,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask,
            is_split_into_words=True,
            return_special_tokens_mask=True,
            add_special_tokens=False,
        )

        # print(len(tokenized['input_ids']))
        padded_length = len(tokenized['input_ids'][0])
        all_segment_ids = [ x + [0] * (padded_length-len(x)) for x in all_segment_ids]
        tokenized['token_type_ids'] = all_segment_ids
        tokenized['label'] = nsp_labels
        tokenized['context_texts'] = context_texts
        tokenized['response_texts'] = response_texts
        # import pdb;pdb.set_trace()            
        # return processed
        to_remove_cols = ['context_string', 'adv_gen_neg_responses_t1', 'adv_gen_neg_responses_k1', 'positive_responses_seqgen', 'adv_gen_neg_responses_k1_seqgen', 'bm25_sampled', 'sentenceBERT_sampled', 'random_sampled']
        for colrname in to_remove_cols:
            if colrname in examples:
                del examples[colrname]

        return tokenized

        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [label_to_id[l] for l in examples["label"]]
        # return result


    # eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.validation_file is not None:
        data_files = {}
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
        kwargs = {'split':'validation'}
        datasets = datasets.map(preprocess_function_test, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)
        eval_dataset = datasets["validation"]
        #     data_files["validation"] = data_args.validation_file
    # import pdb;pdb.set_trace()
    # if data_args.task_name is not None:


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.



    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

        # test_datasets = [test_dataset]
        # for test_dataset, task in zip(test_datasets, tasks):
        #     test_result = trainer.evaluate(eval_dataset=test_dataset)

        #     output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
        #     if trainer.is_world_process_zero():
        #         with open(output_test_file, "w") as writer:
        #             logger.info(f"***** Test results {task} *****")
        #             for key, value in test_result.items():
        #                 logger.info(f"  {key} = {value}")
        #                 writer.write(f"{key} = {value}\n")

####### For test

    if data_args.test_file is not None:
        data_files = {}
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
        kwargs = {'split':'test'}
        datasets = datasets.map(preprocess_function_test, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)
        
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]


    test_results = {}
    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # test_dataset.remove_columns_("label")
            test_dataset_orig = copy.copy(test_dataset)
            test_result = trainer.evaluate(eval_dataset=test_dataset)
            print('test ', test_result)
            output_testmetric_file = os.path.join(training_args.output_dir, f"test_metric_results_{task}.txt")
            # if trainer.is_world_process_zero():
            with open(output_testmetric_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                for key, value in test_result.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            scores =  predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, f"test_outputs_{task}.csv")
            csvwriter = csv.writer(open(output_test_file,'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    csvwriter.writerow(['index','context','response', 'label rev', 'score','prediction','is-correct'])
                    for index, item in enumerate(predictions):
                        test_dp = test_dataset_orig[index]
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            label = test_dp['label']
                            is_correct = (label ==item)
                            result_row = [index,test_dp['context_texts'],test_dp['response_texts'],test_dp['label'],scores[index],item,is_correct]
                            result_row = [str(elem) for elem in result_row]
                            csvwriter.writerow(result_row)
                            # writer.write(f"{index}\t{test_dp['context_texts']}\t{test_dp['response_texts']}\t{test_dp['label']}\t{item}\t{is_correct}\n")
            test_results.update(test_result)
    adv_test_results = test_results.copy()
    # return eval_results

    #now on random
    data_args.type_negative_test = 'random_negative_responses'

    if data_args.test_file is not None:
        data_files = {}
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
        kwargs = {'split':'test'}
        datasets = datasets.map(preprocess_function_test, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)
        
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]


    test_results = {}
    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # test_dataset.remove_columns_("label")
            test_dataset_orig = copy.copy(test_dataset)
            test_result = trainer.evaluate(eval_dataset=test_dataset)
            print('test ', test_result)
            output_testmetric_file = os.path.join(training_args.output_dir, f"random_test_metric_results_{task}.txt")
            # if trainer.is_world_process_zero():
            with open(output_testmetric_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                for key, value in test_result.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            scores =  predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, f"random_test_outputs_{task}.csv")
            csvwriter = csv.writer(open(output_test_file,'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    csvwriter.writerow(['index','context','response', 'label rev', 'score','prediction','is-correct'])
                    for index, item in enumerate(predictions):
                        test_dp = test_dataset_orig[index]
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            label = test_dp['label']
                            is_correct = (label ==item)
                            result_row = [index,test_dp['context_texts'],test_dp['response_texts'],test_dp['label'],scores[index],item,is_correct]
                            result_row = [str(elem) for elem in result_row]
                            csvwriter.writerow(result_row)
                            # writer.write(f"{index}\t{test_dp['context_texts']}\t{test_dp['response_texts']}\t{test_dp['label']}\t{item}\t{is_correct}\n")
            test_results.update(test_result)
    return adv_test_results


def get_model_tokenizer_targetcoherence(model_name_or_path="./targetcoherence/tmp/test-t2neg_context_response_neg_shortcircuit/"):
        # Setup logging
    training_args = TrainingArguments(output_dir='ee',do_predict=True, overwrite_output_dir=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
        # level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    training_args.disable_tqdm =True
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):
    #     # transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Labels

    label_list = [0,1]
    num_labels = 2
    is_regression = False
    config_name = None
    tokenizer_name = None
 
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,
        # cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        # cache_dir=model_args.cache_dir,
        # padding=False,
        # use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        # config=config
    )

    special_tokens_dict = {'additional_special_tokens': ['<eot>']}
    # special_tokens_dict = {'additional_special_tokens': ['<speaker1>','<speaker2>']}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator,
    )

    model = model.to(device)

    return (model, tokenizer, training_args, trainer)


def get_string_text(tokens_a, tokens_b, tokens_target, tokenizer, max_seq_length=256):
    if tokenizer.pad_token in tokens_target:
        tokens_target = tokens_target[:tokens_target.index(tokenizer.pad_token)]
    if tokenizer.pad_token in tokens_a:
        tokens_a = tokens_a[:tokens_a.index(tokenizer.pad_token)]
    if tokenizer.pad_token in tokens_b:
        tokens_b = tokens_b[:tokens_b.index(tokenizer.pad_token)]

    max_num_tokens = max_seq_length - 4
    total_length = len(tokens_a) + len(tokens_b) + len(tokens_target)
    if total_length > max_num_tokens:
        len_b = len(tokens_b) + len(tokens_target)
        if len_b>max_num_tokens:
            tokens_b = tokens_b[:100]
            len_b = len(tokens_b) + len(tokens_target)
        a_begin = abs(max_num_tokens - len_b)
        tokens_a = tokens_a[-a_begin:]
    try:
        assert len(tokens_a) + len(tokens_b) + len(tokens_target) <= max_num_tokens
        assert len(tokens_a) >= 1
    except:
        import pdb;pdb.set_trace()
        print('some problem with preproc')
    #assert len(tokens_b) >= 1

    tokens = []
    segment_ids = []
    tokens.append(tokenizer.cls_token)
    segment_ids.append(0)

    for token in tokens_target:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append(tokenizer.sep_token)
    segment_ids.append(1)

    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append(tokenizer.sep_token)
    segment_ids.append(0)

    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append(tokenizer.sep_token)
    segment_ids.append(1)

    return tokens, segment_ids

def preprocess_function(examples):
    # Tokenize the texts
    processed = []
    items = []
    keys = list(examples.keys())
    # print(examples['context'])
    for i in range(len(examples[keys[0]])):
        ex = {}
        for k in keys:
            ex[k] = examples[k][i]
        items.append(ex)
    # print(len(items))
    all_segment_ids = []
    all_texts = []
    nsp_labels = []
    # speaker1, speaker2 = "<speaker1>", "<speaker2>"
    speaker1, speaker2 = "<eot>", "<eot>"
    data_with_less_negs = 0

    for example in items:
        if example['type'] in data_args.types_to_avoid:
            continue
        if type(example['context']) is list:
            context_arr = [line for line in example['context'] if len(line) > 0 and not line.isspace()]
            context_arr = [speaker2 + ' ' + s if (len(context_arr) - i + 1) % 2 else speaker1 + ' ' + s for i, s in
                           enumerate(context_arr)]
            context = ' '.join(context_arr)
        else:
            context = ' [context] ' + example['context']

        target = example['target']
        if len(target) < 2:
            continue
        target_tokens = tokenizer.tokenize(target)

        tokens_a = context
        tokens_a = tokenizer.tokenize(tokens_a)

        response = example['response']
        if data_args.dontconcattarget is False:
            response = response + ' ' + target
        tokens_b = response
        tokens_b = tokenizer.tokenize(' [response] : ' + tokens_b)#, max_length=data_args.max_seq_length - 5)
        tokens, segment_ids = get_string_text(tokens_a, tokens_b, target_tokens)
        all_texts.append(tokens)
        all_segment_ids.append(segment_ids)
        nsp_labels.append(example['label'])


        # print(len(all_texts), print(range(len(positive_responses), len(example[data_args.type_negative_extra]))))
    # print(data_with_less_negs, ' data_with_less_negs')
    tokenized = tokenizer.batch_encode_plus(
        all_texts,
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # token_type_ids=segment_ids,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask,
        is_split_into_words=True,
        return_special_tokens_mask=True,
        add_special_tokens=False,
    )

    # print(len(tokenized['input_ids']))
    padded_length = len(tokenized['input_ids'][0])
    all_segment_ids = [ x + [0] * (padded_length-len(x)) for x in all_segment_ids]
    tokenized['token_type_ids'] = all_segment_ids
    tokenized['label'] = nsp_labels
    # import pdb;pdb.set_trace()            
    # return processed

    # for k,v in tokenized.items():
    #     print(k, len(v))

    to_remove_cols = ['context_string', 'adv_gen_neg_responses_t1', 'adv_gen_neg_responses_t1_masked', 'adv_gen_neg_responses_k1', 'positive_responses_seqgen', 'adv_gen_neg_responses_k1_seqgen', 'bm25_sampled', 'sentenceBERT_sampled', 'semihard_sampled', 'random_sampled', 'extraction_span', 'backtranslation', 'augmentation', 'augmentation_span', 'noise']
    for colrname in to_remove_cols:
        if colrname in examples:
            del examples[colrname]


    return tokenized



def preprocess_function_test(examples):
    # Tokenize the texts
    processed = []
    items = []
    keys = list(examples.keys())
    # print(examples['context'])
    for i in range(len(examples[keys[0]])):
        ex = {}
        for k in keys:
            ex[k] = examples[k][i]
        items.append(ex)
    # print(len(items))
    all_segment_ids = []
    all_texts = []
    nsp_labels = []
    context_texts = []
    response_texts = []

    speaker1, speaker2 = "<speaker1>", "<speaker2>"
    speaker1, speaker2 = "<eot>", "<eot>"

    for example in items:
        context_arr = [line for line in example['context'] if len(line) > 0 and not line.isspace()]
        context_arr = [speaker2 + ' ' + s if (len(context_arr) - i+1) % 2 else speaker1 + ' ' + s for i, s in enumerate(context_arr)]
        context = ' '.join(context_arr).lower()

        target = example['target'].lower()
        if len(target)<2 and len(items)>4:
            continue
        target_tokens = tokenizer.tokenize(target)
        context_texts.append(target + ' [sep] ' + context)
        tokens_a = context
        tokens_a = tokenizer.tokenize(tokens_a)

        response = example['response'].lower()
        tokens_b = response
        response_texts.append(response)
        tokens_b = tokenizer.tokenize(speaker2 + ' ' + tokens_b, max_length=data_args.max_seq_length-5)
        tokens, segment_ids = get_string_text(tokens_a, tokens_b, target_tokens)
        all_texts.append(tokens)
        all_segment_ids.append(segment_ids)
        nsp_labels.append(example['label'])


    # import pdb;pdb.set_trace()            
    tokenized = tokenizer.batch_encode_plus(
        all_texts,
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # token_type_ids=segment_ids,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask,
        is_split_into_words=True,
        return_special_tokens_mask=True,
        add_special_tokens=False,
    )

    # print(len(tokenized['input_ids']))
    padded_length = len(tokenized['input_ids'][0])
    all_segment_ids = [ x + [0] * (padded_length-len(x)) for x in all_segment_ids]
    tokenized['token_type_ids'] = all_segment_ids
    tokenized['label'] = nsp_labels
    tokenized['context_texts'] = context_texts
    tokenized['response_texts'] = response_texts
    # import pdb;pdb.set_trace()            
    # return processed
    to_remove_cols = ['context_string', 'adv_gen_neg_responses_t1', 'adv_gen_neg_responses_k1', 'positive_responses_seqgen', 'adv_gen_neg_responses_k1_seqgen', 'bm25_sampled', 'sentenceBERT_sampled', 'random_sampled']
    for colrname in to_remove_cols:
        if colrname in examples:
            del examples[colrname]

    return tokenized



def predict_test_file(model, tokenizer, training_args, trainer):
    logger.info("*** Test ***")

    data_files = {}
    data_files["test"] = 'test_cf3.json'
    extension = 'json'
    datasets = load_dataset(extension, data_files=data_files)
    kwargs = {'split':'test'}
    datasets = datasets.map(preprocess_function_test, remove_columns=columns_to_remove, batched=True)
    
    test_dataset = datasets[ "test"]

    test_datasets = [test_dataset]

    for test_dataset, task in zip(test_datasets, ['']):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        # test_dataset.remove_columns_("label")
        test_dataset_orig = copy.copy(test_dataset)
        test_result = trainer.evaluate(eval_dataset=test_dataset)
        print('test ', test_result)
        output_testmetric_file = os.path.join(training_args.output_dir, f"random_test_metric_results_{task}.txt")
        # if trainer.is_world_process_zero():
        with open(output_testmetric_file, "w") as writer:
            logger.info(f"***** Test results {task} *****")
            for key, value in test_result.items():
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        scores =  predictions
        predictions = np.argmax(predictions, axis=1)
        output_test_file = os.path.join(training_args.output_dir, f"random_test_outputs_{task}.csv")
        csvwriter = csv.writer(open(output_test_file,'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                csvwriter.writerow(['index','context','response', 'label rev', 'score','prediction','is-correct'])
                for index, item in enumerate(predictions):
                    test_dp = test_dataset_orig[index]
                    item = label_list[item]
                    label = test_dp['label']
                    is_correct = (label ==item)
                    result_row = [index,test_dp['context_texts'],test_dp['response_texts'],test_dp['label'],scores[index],item,is_correct]
                    result_row = [str(elem) for elem in result_row]
                    csvwriter.writerow(result_row)


def preprocess_function_datapoint(examples, tokenizer, dontconcattarget=False):
    # Tokenize the texts
    processed = []
    items = []
    keys = list(examples[0].keys())
    # print(len(items))
    all_segment_ids = []
    all_texts = []
    nsp_labels = []
    context_texts = []
    response_texts = []

    speaker1, speaker2 = "<speaker1>", "<speaker2>"
    speaker1, speaker2 = " [eot] ", " [eot] "
    max_seq_length = 256 

    for i, example in enumerate(examples):
        if type(example['context']) is list:
            context_arr = [line for line in example['context'] if len(line) > 0 and not line.isspace()]
            # context_arr = [speaker2 + ' ' + s if (len(context_arr) - i + 1) % 2 else speaker1 + ' ' + s for i, s in
            #                enumerate(context_arr)]
            context = ' [context] ' + ' '.join(context_arr).strip()
        else:
            context = ' [context] ' + example['context'].strip()

        target = example['target']

        target_tokens = tokenizer.tokenize(target)

        tokens_a = context
        tokens_a = tokenizer.tokenize(tokens_a)

        response = example['response']
        if dontconcattarget is False:
            response = response + ' ' + target
        tokens_b = response
        tokens_b = tokenizer.tokenize(' [response] : ' + tokens_b)#, max_length=max_seq_length - 5)
        tokens, segment_ids = get_string_text(tokens_a, tokens_b, target_tokens, tokenizer)#, max_seq_length=max_seq_length)
        all_texts.append(tokens)
        all_segment_ids.append(segment_ids)
        nsp_labels.append(example.get('label', 0))

    padding = "max_length"
    max_length = max_seq_length
    # import pdb;pdb.set_trace()            
    tokenized = tokenizer.batch_encode_plus(
        all_texts,
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # token_type_ids=segment_ids,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask,
        is_split_into_words=True,
        return_special_tokens_mask=True,
        add_special_tokens=False,
    )

    # print(len(tokenized['input_ids']))
    padded_length = max_length#len(tokenized['input_ids'])
    all_segment_ids = [ x + [0] * (padded_length-len(x)) for x in all_segment_ids]
    tokenized['token_type_ids'] = all_segment_ids
    # tokenized['label'] = nsp_labels
    # tokenized['context_texts'] = context_texts
    # tokenized['response_texts'] = response_texts
    # import pdb;pdb.set_trace()            
    # return processed
    to_remove_cols = ['context_string', 'adv_gen_neg_responses_t1', 'adv_gen_neg_responses_k1', 'positive_responses_seqgen', 'adv_gen_neg_responses_k1_seqgen', 'bm25_sampled', 'sentenceBERT_sampled', 'random_sampled']
    for colrname in to_remove_cols:
        if colrname in examples:
            del examples[colrname]

    return tokenized


def create_data_loader(tokenized_eval_dataset, batch_size):
    return DataLoader(
        tokenized_eval_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=default_data_collator

    )

def predict_testpoint_targetcoherence(model_objects, examples,dontconcattarget=False, batch_size=20, verbose=False):
    model, tokenizer, training_args, trainer = model_objects
    datapoint_dict = preprocess_function_datapoint(examples, tokenizer, dontconcattarget=dontconcattarget)
    # import pdb;pdb.set_trace()
    test_dataset = Dataset.from_dict(datapoint_dict)
    test_data_loader = create_data_loader(test_dataset, batch_size)

    all_scores = []
    for d in test_data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        preds_logits = outputs[0].tolist()
        # print(preds_logits)
        preds = np.argmax(preds_logits, axis=1)
        softmax_l1 = softmax(preds_logits, axis=1).tolist()
        tc_score = [x[1] for x in softmax_l1]
        # all_scores += softmax_l1
        all_scores+=tc_score

    # test_dataset_orig = copy.copy(test_dataset)
    # predictions = trainer.predict(test_dataset=test_dataset).predictions
    # scores =  predictions
    # scores = softmax(scores, axis=1)
    # predictions = np.argmax(predictions, axis=1)
    # all_scores = []
    # if trainer.is_world_process_zero():
    #     for index, item in enumerate(predictions):
    #         test_dp = test_dataset_orig[index]
    #         # item = label_list[item]
    #         # label = test_dp['label']
    #         # is_correct = (label ==item)
    #         if verbose:
    #             result_row = [index, test_dp['context_texts'], test_dp['response_texts'], test_dp['label'],
    #                           scores[index]]
    #             print(result_row)
    #         # result_row = [str(elem) for elem in result_row]
    #         # csvwriter.writerow(result_row)
    #         all_scores.append(scores[index][1])

    return all_scores

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # model_objects = get_model_tokenizer_targetcoherence(model_name_or_path='../data_prep/otters/tmp/test-alv2-t1neg_response_target_specific/')
    model_objects = get_model_tokenizer_targetcoherence(model_name_or_path='../data_prep/otters/tmp/test-alv2-withneg_response_target_specific/')
    # predict_test_file(model, tokenizer, training_args, trainer)
    examples = [{'context': 'i enjoy staring up at the sky.', 'response': "I like stargazing outside with my pet.",
                 'target': 'i like to spend a lot of my free time with my pet.'}, \
                 {'context': 'i enjoy staring up at the sky.', 'response': "I like stargazing outside.",
                 'target': 'i like to spend a lot of my free time with my pet.'}, \
                 {'context': 'i enjoy staring up at the sky.', 'response': "My pet likes food.",
                 'target': 'i like to spend a lot of my free time with my pet.'}, \
                {'context': ["excuse me , ma'am . can you tell me where the nearest postoffice is ?", "of course . go straight ahead . turn right at the next street .\
          you 'll see a tall , yellow building . the post office is on the first floor .",
                             'do you mean that i go that way for one block , then turn right ?',
                             'yes , you are right .', 'is it far ?'],
                 'response': "no , it 's only about five minutes walk .",
                 'target': "lets go for talk"}]
    all_scores = predict_testpoint_targetcoherence(model_objects, examples, dontconcattarget=True)
    print(all_scores)
    # main()

#python run_targetcoherence.py  --test_file /home/ubuntu/Code/negaug/dataset/test.json --output_dir tmp/test-e1_logs  --per_device_eval_batch_size 12  --type_negative_test random_negative_responses --overwrite_cache --do_predict --overwrite_output_dir
