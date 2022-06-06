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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import random
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import BlenderbotSmallTokenizer, BlenderbotTokenizer, BlenderbotForConditionalGeneration

from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


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
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
            if extension == 'jsonl':extension = 'json'
            raw_traindatasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        if data_args.validation_file is not None:
            data_files = {}
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
            if extension == 'jsonl':extension = 'json'
            raw_validdatasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        if extension == 'jsonl':
            extension = 'json'
        # extension = 'text'
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)#, column_names=["index", "context", "target", "output"])
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if 'blender' not in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model = AutoModelForSeq2SeqLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    else:
        model = BlenderbotForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        tokenizer = BlenderbotTokenizer.from_pretrained(model_args.model_name_or_path)

    model.resize_token_embeddings(len(tokenizer))

    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
#     if training_args.do_train:
#         column_names = raw_datasets["train"].column_names
#     elif training_args.do_eval:
#         column_names = raw_datasets["validation"].column_names
#     elif training_args.do_predict:
#         column_names = raw_datasets["test"].column_names
#     else:
#         logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
#         return

    # Get the column names for input/target.
#     dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
#     if data_args.text_column is None:
#         text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
#     else:
#         text_column = data_args.text_column
#         if text_column not in column_names:
#             raise ValueError(
#                 f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
#             )
#     if data_args.summary_column is None:
#         summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
#     else:
#         summary_column = data_args.summary_column
#         if summary_column not in column_names:
#             raise ValueError(
#                 f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
#             )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function_summarization(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def preprocess_function(examples):
#         import pdb;pdb.set_trace()

#         inputs = [prefix + inp for inp in inputs]
        items = []
        keys = list(examples.keys())
            # print(examples['context'])
        for i in range(len(examples[keys[0]])):
            ex = {}
            for k in keys:
                ex[k] = examples[k][i]
            items.append(ex)
        inputs, targets = [], []
        
        for example in items:
            context = example['history']
            context_str = ' </s> <s> '.join(context)
            knowledge_list = example['knowledge']
            knowledge_list = ['title: '+x.replace('__knowledge__', 'content:') for x in knowledge_list]
            knowledge_used = knowledge_list[0]
            if 'no_passages_used' in knowledge_used:
                knowledge_used = ''
            knowledge_random = random.choice(knowledge_list)
            thresh_val = random.random()
            if thresh_val>0.35:
                knowledge_selected = knowledge_used
            elif thresh_val>0.2:
                knowledge_selected = knowledge_random
            else:
                knowledge_selected = ''
#             knowledge_selected = knowledge_used
            input_ex =  knowledge_selected + ' </s> context: <s> ' + context_str
            target_ex = example['response']
            inputs.append(input_ex)
            targets.append(target_ex)
        

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def find_sub_list(sl, lists):
        results = []
        sll = len(sl)
        for l in lists:
            result = None
            for ind in (i for i, e in enumerate(l) if e == sl[0]):
                if l[ind:ind + sll] == sl:
                    result = (ind + sll - 1)
            results.append(result)

        return results

    # greetings = [
    #     [72, 3128, 257, 1256, 286, 4813, 13, 685, 22866, 60, 1312, 1560, 14532, 319, 3800, 13, 685, 26209, 60, 1058,
    #      3878, 11, 857, 326, 651, 345, 257, 1256, 286, 17308, 30, 314, 1842, 3249, 649, 1466, 13, 1279, 13492, 29, 220,
    #      50256, 50257],
    #     [685, 22866, 60, 1312, 1560, 14532, 319, 3800, 13, 685, 26209, 60, 1058, 3878, 11, 857, 326, 651, 345, 257,
    #      1256, 286, 17308, 30, 314, 1842, 3249, 649, 1466, 13, 1279, 13492, 29, 220, 50256, 50257], ]
    # find_sub_list([685, 26209, 60, 1058], greetings)

    def preprocess_function_otters(examples):
        items = []
        keys = list(examples.keys())
        # print(examples['context'])
        for i in range(len(examples[keys[0]])):
            ex = {}
            for k in keys:
                ex[k] = examples[k][i]
            items.append(ex)
        inputs, targets = [], []

        for example in items:
            if 'da_string' in example and 'State' not in example['da_string']: continue
            if 'tc_score' in example and example['tc_score'][1]<0.7: continue
            if 'original_response' in example:
                example['response'] = example['original_response']
            # print(example)
            # import pdb;pdb.set_trace()
            context = example['context']
            if type(context) == str:
                context = [context]
            context = context[-2:]
            context_str = ' [eot] '.join(context)
            context_words = context_str.split()
            if len(context_words) > data_args.max_source_length - 100:
                context_str = ' '.join(context_words[-150:])
            # knowledge_selected = example['target']
            # target_ex = example['response']
            # input_ex = knowledge_selected + ' [context] ' + context_str + ' [response] : ' + target_ex + ' <eor> ' + tokenizer.eos_token
            target = example['target']
            response = example['response']
            path = example['path']
            # input_ex = knowledge_selected + ' [context] ' + context_str + ' [response] : ' + target_ex + ' <eor> ' + tokenizer.eos_token
            input_ex = '[knowledge clue] ' + path + ' [target] ' + target + ' [context] ' + context_str + ' [response] : ' + response + ' <eor> ' + tokenizer.eos_token
            # input_ex = '[target] ' + target + ' [context] ' + context_str + ' [knowledge clue] ' + path +  ' [response] : ' + response + ' <eor> ' + tokenizer.eos_token
            # print(input_ex)
            # import pdb;pdb.set_trace()
            inputs.append(input_ex)
            # targets.append(target_ex)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        # with tokenizer.as_target_tokenizer():
        #     labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        model_inputs["labels"] = model_inputs["input_ids"]

        #to train only over responses
        context_end_indices = find_sub_list(tokenizer.encode(' [response] :'), model_inputs["input_ids"])
        new_labels = []
        for i, lab in enumerate(model_inputs["input_ids"]):
            if context_end_indices[i] is None:
                new_labels.append(lab)
            else:
                context_end_index = context_end_indices[i]
                new_label = [-100]*(context_end_index+1) + lab[context_end_index+1:]
                new_labels.append(new_label)
        model_inputs["labels"] = new_labels

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]

        for k, v in model_inputs.items():
            if k not in ['input_ids', 'attention_mask', 'labels']:
                model_inputs.pop(k, None)


        return model_inputs


    # column_names = ['context', 'da_raw', 'da_string', 'is_targetatend', 'original_response', 'response', 'overlap_tr', 'tc_score', 'target']
    column_names = ['context', 'target', 'response', 'path', 'score_path', 'type', 'path_headentity', 'path_tailentity', 'entities_inpath', 'original_response', 'tc_score', 'da_string', 'da_raw']
    if training_args.do_train:
        if "train" not in raw_traindatasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_traindatasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_otters,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
#                 remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    column_names = ['context', 'target', 'response', 'path', 'score_path', 'type', 'path_headentity', 'path_tailentity', 'entities_inpath']

    # if training_args.do_eval:
    #     max_target_length = data_args.val_max_target_length
    #     if "validation" not in raw_validdatasets:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = raw_validdatasets["validation"]
    #     if data_args.max_train_samples is not None:
    #         train_dataset = train_dataset.select(range(data_args.max_train_samples))
    #     with training_args.main_process_first(desc="train dataset map pre-processing"):
    #         train_dataset = train_dataset.map(
    #             preprocess_function_otters,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on validation dataset",
    #         )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        # if "validation" not in raw_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        # eval_dataset = raw_datasets["validation"]
        if "validation" not in raw_validdatasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_validdatasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_otters,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_otters,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

# python train_pathddgpt.py --train_file augfp4_train_agendaseqwsrl_goldkeywords_tcfiltered0.75_duppathremoved_len10.jsonl --validation_file validddpaths.jsonl --model_name_or_path gpt2 --output_dir alv2-tcfiltered0.75_duppathremoved_len10-v1 --num_train_epochs 2  --do_train  --do_eval  --overwrite_output_dir --eval_steps 200  --evaluation_strategy steps --fp16 --logging_steps 100 --pad_to_max_length --per_device_train_batch_size 12 --per_device_eval_batch_size 16  --gradient_accumulation_steps 4
