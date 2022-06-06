'''
script based on code adapted from 
https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation
'''


import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
import csv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers.generation_utils import top_k_top_p_filtering
# from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
import pandas as pd
import jsonlines
random.seed(1234)
def clean_generation(results):
    if type(results)==list:
        results = results[0]
    results = results.split('[response] : ')[-1]
    eor_ind = results.find('<eor')
    final = results[:eor_ind - 1].strip()
    
    return final

def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def select_pathspercontext(data):
    tc_dict = dict()
    for row in data:
        target = row['target']
        context = row['context']
        path = row['path']
        if (target,context) not in tc_dict:
            tc_dict[(target,context)] = []
        tc_dict[(target, context)].append(row)
        # references = row['outputs']

    final_data = []
    num_rows = 0
    for tc in tc_dict.keys():
        target, context = tc
        gen_paths = tc_dict[(target, context)]
        num_paths = len(gen_paths)
        required_paths = len(gen_paths[0]['outputs'])
        selected_rows = random.sample(gen_paths, min(required_paths, num_paths))
        selected_paths = [x['path'] for x in selected_rows]
        if len(selected_rows)>0:
            newrow = dict(selected_rows[0])
            newrow['paths'] = selected_paths
            final_data.append(newrow)
            num_rows+=len(selected_paths)
        # final_data+=selected_rows
    # import pdb;pdb.set_trace()
    print('len final_data', len(final_data), 'rows ', num_rows)
    return final_data

def get_model_suffix(model_string):
    modelsuffix = ''
    cands = args.model_string.split('/')
    for cand in cands:
        if cand!='':
            modelsuffix = cand

    return modelsuffix

def main_file(args):
    test_data = get_json_lines(args.test_file)
    test_data = select_pathspercontext(test_data)
    # if args.topnrows is not None:
    #     test_data = test_data.head(args.topnrows)
    # test_data.columns = ['index', 'context', 'target', 'output', 'domain']
    # test_data = test_data.groupby(['context', 'target']).apply(lambda x: [list(x['output']), list(x['domain'])]).apply(pd.Series).reset_index()
    columns = ['context', 'target',  'references', 'all_puretext', 'all_modelgen', 'paths']
    model_suffix = get_model_suffix(args.model_string)+'-pk'+str(args.precondition_topk)+'-cl'+str(args.condition_lambda)+'.csv'
    if args.do_sample is False:
        model_suffix = model_suffix.replace('.csv', '_nosample.csv')
    csvfile = open('ottersdata/'+ args.test_file.split('/')[-1].split('.')[0]+model_suffix, 'w')
        # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(columns)

    with open('/'.join(args.ckpt.split('/')[:-1])+'/dataset_info', 'rb') as rf:
        dataset_info = pickle.load(rf)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_string)
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0]
    model = GPT2LMHeadModel.from_pretrained(args.model_string, return_dict=True).to(args.device)
    model.eval()


    print('model_args', model_args)

    # for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    for index, row in enumerate(tqdm(test_data)):
        # domain = row['domain']
        target = row['target']
        context = row['context']
        if type(context) is str:
            context = [context]
        references = row['outputs']
        num_generation = len(references)#len(row['paths'])#len(references)
        all_puretext = []
        all_modelgen = []
        all_paths = []
        for n in range(num_generation):
            path = row['path']
            if 'paths' in row:
                path = row['paths'][n]
            print(context, '--- with target: ', target, references, 'path: ', path)
            all_paths.append(path)
            input_text = '[knowledge clue] ' + path + ' [target] ' + target + ' [context] ' + ' [eot] '.join(context) + ' [response] :'
            # input_text = target + ' [context] ' + ' [eot] '.join(context) + ' [response] :'
            # input_text = '[target] ' + target + ' [context] ' + ' [eot] '.join(context) + ' [response] :' #for withnopath models
            results_puregen = predict_formality(model,
                            tokenizer,
                            None,
                            [input_text],
                            dataset_info,
                            precondition_topk=args.precondition_topk,
                            do_sample=args.do_sample,
                            length_cutoff=args.length_cutoff,
                            condition_lambda=0,
                            device=args.device,
                            verbose=False,)
            results_puregen = clean_generation(results_puregen)
            all_puretext.append(results_puregen)
            # print(results_puregen, '--no classifier')
        

        csvwriter.writerow([context, target, references, all_puretext, all_modelgen, all_paths])
        csvfile.flush()
        # import pdb; pdb.set_trace()

    csvfile.close()


def predict_formality(model, tokenizer, conditioning_model, input_text, dataset_info, precondition_topk=200, do_sample=True, length_cutoff=512, condition_lambda=1.0, device='cuda', verbose=True):
    with torch.no_grad():
        batch_size = len(input_text)
        # assumes initially all same length.
        encoded_input = [tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)

        # input_ids = torch.LongTensor([[65000]]).to(device)
        input_ids = encoded_input.to(device)
        cur_len = 1
        max_length = length_cutoff
        min_length = 0
        temperature = 0.7#1.0
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = [[65000]]
        pad_token_id = 65000
        eos_token_id = 0
        effective_batch_size = batch_size
        attention_mask = encoded_input.new_ones(encoded_input.shape)
        use_cache = True
        # model_specific_kwargs = {'encoder_outputs': model.get_encoder()(encoded_input, attention_mask=attention_mask)}
        model_specific_kwargs = {'encoder_outputs': model(encoded_input, attention_mask=attention_mask)}

        output = _generate_no_beam_search(model, tokenizer,
                                        conditioning_model,
                                        condition_lambda,
                                        precondition_topk,
                                        encoded_input,
                                        input_ids,
                                        cur_len,
                                        max_length,
                                        min_length,
                                        do_sample,
                                        temperature,
                                        top_k,
                                        top_p,
                                        repetition_penalty,
                                        no_repeat_ngram_size,
                                        bad_words_ids,
                                        pad_token_id,
                                        eos_token_id,
                                        batch_size,
                                        attention_mask,
                                        use_cache,
                                        model_specific_kwargs,
                                          verbose=verbose,)

        return [tokenizer.decode(s[:]) for s in output] # 1: to delete the pad token


# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search(
        model, tokenizer,
        conditioning_model,
        condition_lambda,
        precondition_topk,
        encoded_input,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
        verbose=True,
):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            # model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs)
            model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache)
            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # scores = model.postprocess_next_token_scores(
            scores = postprocess_next_token_scores(
                scores=next_token_logits,input_ids=input_ids,no_repeat_ngram_size=no_repeat_ngram_size,bad_words_ids=bad_words_ids,
                cur_len=cur_len,min_length=min_length,max_length=max_length,eos_token_id=eos_token_id,repetition_penalty=repetition_penalty,batch_size=batch_size,num_beams=1,)

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems
            tt = tokenizer
            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # batch x topk
            input_ids_suffix = input_ids
            if condition_lambda>0:
                target_idx = (input_ids[0] == 16793).nonzero(as_tuple=True)[-1].tolist()
                if len(target_idx)>0 and input_ids[0][target_idx[0]+1].item()==60:
                    input_ids_suffix = input_ids[:,target_idx[0]+2:]
                # print(input_ids_suffix)
                # import pdb;pdb.set_trace()
            tplus1_candidates = torch.cat([input_ids_suffix.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, 1:] # batch x topk x seq+1, with pad dropped
            expanded_lengths = torch.LongTensor([[cur_len for _ in range(precondition_topk)] for _ in range(batch_size)]).to(scores.device)
            input_lengths = torch.LongTensor([tplus1_candidates.shape[2] for _ in range(precondition_topk)]).to(scores.device)

            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else:                
                condition_logits = conditioning_model(tplus1_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    # expanded_lengths.flatten(0, 1), # batch*topk
                                                    input_lengths,
                                                    None,
                                                    None,
                                                    None)
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1)[:, :, -1] # batch x topk of last formality pred
                # condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs
                # print(condition_logits)
            full_logits = top_logits + condition_lambda * condition_logits
            scores = F.softmax(full_logits, dim=-1)

            top_probs = F.softmax(top_logits, dim=-1)
            condition_logits_prob = F.softmax(condition_logits, dim=-1)
            # full_probs = top_probs + condition_lambda * condition_logits_prob
            # scores = F.normalize(full_probs,dim=-1, p=1)

            if do_sample:
                scores = scores / temperature
                # scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                topchosen_token_indice = torch.multinomial(scores, num_samples=1).squeeze(1)
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), topchosen_token_indice]
            else:
                # Greedy decoding
                topchosen_token_indice = torch.argmax(scores, dim=-1)
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(scores, dim=-1)]

            if verbose and condition_lambda>0:
                print([(i, tokenizer.decode([z]), round(x,4),round(y,4), round(s,4)) for i, (x,y,s,z) in enumerate(zip(top_probs[0].tolist(), condition_logits_prob[0].tolist(), scores[0].tolist(), top_indices[0].tolist()))])
                # print([tokenizer.decode([x]) for x in top_indices[0]])
                print(tokenizer.decode(input_ids[0]))
                print(next_token[0], tokenizer.decode([next_token]), topchosen_token_indice[0].item())

            tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if next_token[0]==tokenizer.eos_token_id:
                break
            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores

def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens

def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=False)
    parser.add_argument('--dataset_info', type=str, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='microsoft/DialoGPT-medium')

    parser.add_argument('--input_text', type=str, default='abc', help='text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=60, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--topnrows', type=int, help='first n rows')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--test_file', type=str, default='in_domain_test_out.csv')
    # parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()
    target = "my mom was a british ballerina."
    context = ["i am relocating for a job."]
    args.input_text = target + ' ' + ' '.join(context) + ' '
    args.input_text = target + ' [context] ' + ' [eot] '.join(context) + ' [response] :'
    print('args', args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main_file(args)

