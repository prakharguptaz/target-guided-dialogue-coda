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
import torch
from dataclasses import dataclass, field
from typing import Optional
import csv
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process


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

logger = logging.getLogger(__name__)


# columns_to_remove = ['id','adversarial_negative_responses', 'random_negative_responses', 'positive_responses', 'context']
columns_to_remove = ['context', 'response', 'target', 'label', 'type', 'index']
# types_to_avoid = ['neg_context_target']

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
        default=256,
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

    type_negative_extra_max: Optional[int] = field(
        default=1000, metadata={"help": "A csv or a json file containing the test data."}
    )

    type_negative_test: Optional[str] = field(
        default='random_negative_responses', metadata={"help": "A csv or a json file containing the test data."}
    )
    types_to_avoid: Optional[str] = field(
        default='', metadata={"help": "semicolon separated filed to avoid in data preparation"}
    )
    dontconcattarget: bool = field(
        default=False, metadata={"help": "concatenate response and target for otters"}
    )
    given_seed: Optional[int] = field(
        default=100, metadata={"help": "A seed."}
    )



    # def __post_init__(self):
    #     if self.task_name is not None:
    #         self.task_name = self.task_name.lower()
    #         if self.task_name not in task_to_keys.keys():
    #             raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
    #     elif self.train_file is None or self.validation_file is None:
    #         raise ValueError("Need either a GLUE task or a training/validation file.")
    #     else:
    #         extension = self.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
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

    training_args.output_dir = training_args.output_dir+(data_args.types_to_avoid)
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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # training_args.save_steps=2000
    # training_args.eval_steps=2000
    # training_args.evaluate_during_training=True
    # training_args.evaluation_strategy = 'epoch'
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(data_args.given_seed)
    # set_seed(training_args.seed)

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
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    elif training_args.do_train and data_args.train_file.endswith(".csv"):
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
            if extension == "jsonl":
                extension = "json"
            # datasets = load_dataset(extension, data_files=data_files)
            # datasets = load_dataset(extension, data_files=data_files, split='train[:9259]')
            datasets = load_dataset(extension, data_files=data_files, split='train[:100%]', cache_dir='./cached')

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
    add_prefix_space = False
    if 'roberta' in model_args.model_name_or_path:
        add_prefix_space = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    if 'roberta' in model_args.model_name_or_path:
        # import pdb;pdb.set_trace()
        model.config.type_vocab_size = 2
        single_emb = model.roberta.embeddings.token_type_embeddings
        model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        model.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

    special_tokens_dict = {'additional_special_tokens': ['<eot>']}
    # special_tokens_dict = {'additional_special_tokens': ['<speaker1>','<speaker2>']}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    # import pdb;pdb.set_trace()
    # Preprocessing the datasets
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
    # padding = True
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

    def get_string_text(tokens_a, tokens_b, tokens_target):
        max_num_tokens = data_args.max_seq_length - 4
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
            print('some problem with preproc')
            return None, None
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
        # speaker1, speaker2 = " [eot] ", " [eot] "
        speaker1, speaker2 = "<eot>", "<eot>"
        data_with_less_negs = 0

        for example in items:
            if example['type'] in data_args.types_to_avoid:
                continue
            if type(example['context']) is list:
                context_arr = [line for line in example['context'] if len(line) > 0 and not line.isspace()]
                context_arr = [speaker2 + ' ' + s if (len(context_arr) - i+1) % 2 else speaker1 + ' ' + s for i, s in enumerate(context_arr)]
                context = ' '.join(context_arr)
            else:
                context = ' [context] ' + example['context']

            target = example['target']
            if len(target)<2:
                continue
            target_tokens = tokenizer.tokenize(target)

            tokens_a = context
            tokens_a = tokenizer.tokenize(tokens_a)

            response = example['response']
            if data_args.dontconcattarget is False:
                response = response + ' ' + target
            tokens_b = response
            tokens_b = tokenizer.tokenize(' [response] : ' + tokens_b, max_length=data_args.max_seq_length-5)
            tokens, segment_ids = get_string_text(tokens_a, tokens_b, target_tokens)
            if tokens == None:
                continue
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
        all_types_labels = []

        speaker1, speaker2 = "<speaker1>", "<speaker2>"
        speaker1, speaker2 = "<eot>", "<eot>"

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
            context_texts.append(target + ' [context] ' + context)

            tokens_a = context
            tokens_a = tokenizer.tokenize(tokens_a)

            response = example['response']
            if data_args.dontconcattarget is False:
                response = response + ' ' + target
            response_texts.append(response)
            tokens_b = response
            tokens_b = tokenizer.tokenize(' [response] : ' + tokens_b, max_length=data_args.max_seq_length - 5)
            tokens, segment_ids = get_string_text(tokens_a, tokens_b, target_tokens)
            if tokens == None:
                continue
            all_texts.append(tokens)
            all_segment_ids.append(segment_ids)
            nsp_labels.append(example['label'])
            all_types_labels.append(example['type'])

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
        tokenized['type_label'] = all_types_labels

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

    if training_args.do_train:
        datasets = datasets.map(preprocess_function, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)

        train_dataset = datasets
        # train_dataset = datasets["train"]

        print(len(train_dataset), 'len train_dataset')
            # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.validation_file is not None:
        data_files = {}
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "jsonl":
            extension = "json"
        datasets = load_dataset(extension, data_files=data_files)
        kwargs = {'split':'validation'}
        datasets = datasets.map(preprocess_function_test, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)
        eval_dataset = datasets["validation"]
        #     data_files["validation"] = data_args.validation_file
    # import pdb;pdb.set_trace()
    # if data_args.task_name is not None:
    if data_args.test_file is not None:
        data_files = {}
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "jsonl":
            extension = "json"
        datasets = load_dataset(extension, data_files=data_files)
        kwargs = {'split':'test'}
        datasets = datasets.map(preprocess_function_test, remove_columns=columns_to_remove, batched=True, load_from_cache_file=not data_args.overwrite_cache)
        
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]


    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            # return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
            # import pdb;pdb.set_trace()
            tp = [1 if pred == label ==0 else 0 for pred,label in zip(preds, p.label_ids)]
            tn = [1 if pred == label ==1 else 0 for pred,label in zip(preds, p.label_ids)]
            fp = [1 if pred == 0 and label ==1 else 0 for pred,label in zip(preds, p.label_ids)]
            fn = [1 if pred == 1 and label ==0 else 0 for pred,label in zip(preds, p.label_ids)]
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                    'tp': sum(tp), #labels are flipped
                    'tn': sum(tn),
                    'fp': sum(fp),
                    'fn': sum(fn)
                    }


    # Initialize our Trainer
    if training_args.do_train:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator if data_args.pad_to_max_length else None,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
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
            # scores =  predictions
            predictions_tensor = torch.from_numpy(predictions)
            scores = torch.softmax(predictions_tensor, dim=1)[:, 0].tolist()
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, f"test_outputs_{task}.csv")
            # import pdb;pdb.set_trace()
            csvwriter = csv.writer(open(output_test_file,'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            print('writing to ', output_test_file)
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    csvwriter.writerow(['index','context','response', 'label rev', 'score','prediction', 'type_label','is-correct'])
                    for index, item in enumerate(predictions):
                        test_dp = test_dataset_orig[index]
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            label = test_dp['label']
                            is_correct = (label ==item)
                            result_row = [index,test_dp['context_texts'],test_dp['response_texts'],test_dp['label'],scores[index],item, test_dp.get('type_label', ''),is_correct]
                            result_row = [str(elem) for elem in result_row]
                            csvwriter.writerow(result_row)
                            # writer.write(f"{index}\t{test_dp['context_texts']}\t{test_dp['response_texts']}\t{test_dp['label']}\t{item}\t{is_correct}\n")
            test_results.update(test_result)
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

# test python run_dialclassifier.py --model_name_or_path tmp/test-alv2-t1neg_response_target_specific --validation_file ottersnegdata/total_dev_wneg.jsonl ---output_dir tmp/test-alv2-t1neg_response_target_specific  --per_device_eval_batch_size 120 --overwrite_cache   --do_eval  --overwrite_output_dir --load_best_model_at_end --types_to_avoid neg_response_target_specific --metric_for_best_model accuracy
