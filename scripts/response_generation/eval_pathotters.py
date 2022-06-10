import scipy.stats
import numpy as np
import json
import os
import csv
import random
import pandas as pd
from run_bert_coherence import get_model_tokenizer_coherence, predict_testpoint_coherence
from run_targetcoherence import get_model_tokenizer_targetcoherence, predict_testpoint_targetcoherence
import transformers
import logging
from itertools import groupby
from tqdm import tqdm
import textdistance
from collections import defaultdict
import spacy
from ast import literal_eval
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
from sacrebleu.metrics import BLEU, CHRF, TER
import sacrebleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score as nltkbleu
from nltk.tokenize.regexp import regexp_tokenize
from scipy.stats import pearsonr, spearmanr


import bert_score
# import rake_utils
bert_scorer = bert_score.BERTScorer(
        lang="en", batch_size=3, rescale_with_baseline=True
    )
def bertscore_multi_refs_working(bert_scorer, bertscore_cands, bertscore_refs):
    scorer = bert_scorer  # bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=True)
    cands = bertscore_cands  # reader.bertscore_cands
    refs = bertscore_refs  # reader.bertscore_refs
    P_mul, R_mul, F_mul = scorer.score(
        cands,
        refs,
    )
    # print("P_mul, R_mul, F_mul = ", P_mul, R_mul, F_mul)
    # print("P_mul, R_mul, F_mul = ", np.mean(P_mul.data.cpu().numpy()), np.mean(R_mul.data.cpu().numpy()), np.mean(F_mul.data.cpu().numpy()))
    return {
        "bert_prec": np.mean(P_mul.data.cpu().numpy()),
        "bert_rec": np.mean(R_mul.data.cpu().numpy()),
        "bert_f1": np.mean(F_mul.data.cpu().numpy()),
    }


logging.getLogger().setLevel(logging.WARNING)
transformers.logging.set_verbosity(transformers.logging.WARNING)
transformers.logging.get_verbosity()
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
        # level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''
bleu = BLEU()
refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
    ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

# bleu = BLEU()
# bscore = bleu.corpus_score(sys, refs)
# print(bscore)

def analyze_scores(args):
    data = []
    with open(args.input_file) as f:
        for line in f:
            data.append(json.loads(line))
    type0_score = 'c_scores'
    type1_score = 'tc_scores'
    type2_score = 'bleu_scores'
    scores1 = []
    scores2 = []
    corr_scores1 = []
    corr_scores2 = []
    model_list = defaultdict(lambda: defaultdict(list))
    for dp in data:
        is_reference = False
        model_name = None
        domain = dp['domain'][0]
        for k in dp.keys():
            if 'scores' in k:
                model_name = k.split('_')[0]
            if 'references_' in k:
                is_reference = True
        # if domain == 'out_of_domain':
        #     continue
        for k in dp.keys():
            if 'scores' in k:
                model_list[model_name][k].append(dp[k])
        # if not is_reference:
        #     continue

        type2_score_k = model_name + '_' + type2_score
        type1_score_k = model_name + '_' + type1_score
        if type2_score_k in dp.keys():
            scores2.append(dp[type2_score_k])
            if type1_score_k in dp.keys():
                scores1.append(dp[type1_score_k])
                if is_reference:
                    corr_scores1.append(dp[type2_score_k])
                    corr_scores2.append(dp[type1_score_k])

    for model in model_list.keys():
        print(model)
        for typescorelist in model_list[model].keys():
            list_scores = model_list[model][typescorelist]
            # import pdb;pdb.set_trace()
            print(typescorelist, sum(list_scores)/len(list_scores), len(list_scores))


    print(len(corr_scores1), len(corr_scores2), 'scores len')
    corr, s = pearsonr(np.array(corr_scores1), np.array(corr_scores2))
    print('Pearsons score: ', corr, s)

    corr, s = spearmanr(np.array(corr_scores1), np.array(corr_scores2))
    print('Spearman score: ', corr, s)
    # import pdb;pdb.set_trace()


def print_coco_scores(data):
    from cocoevals import CocovalsMeasures
    evals_obj = CocovalsMeasures()
    refs = data['references']
    sys = data['response']
    print(len(refs), len(sys))
    i = 0
    for refsi,sysi in zip(refs,sys):
        i+=1
        for refsij in refsi:
            evals_obj(prediction=sysi, ground_truth=refsij, id=i)
    metrics = evals_obj.get_metric(reset=True)
    print("metrics = ", metrics)
    ###bert-score
    tmp = defaultdict(list)
    for refsi, sysi in zip(refs, sys):
        info = bertscore_multi_refs_working(
            bert_scorer,
            bertscore_cands=[sysi],
            bertscore_refs=[refsi],
        )
        for k,v in info.items():
            tmp[k].append(v)
        # topk_vals = rake_utils.get_hits_topk(candidate=sysi,
        #                                      references=refsi,
        #                                      kset=[1, 2, 3, 5],
        #                                      method='rake')
        # for k,v in topk_vals.items():
        #     tmp[k].append(v)
    for k,vlist in tmp.items():
        metrics[k] = np.mean(vlist)
    print("metrics = ", metrics)


def read_predicted_data(fname):
    # Loads the csv files
    # each row is organized as a dictionary
    import csv
    data = []
    with open(fname, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
    header = data[0]
    rows = data[1:]
    print("header = ", header)
    ret = []
    for i, row in enumerate(rows):
        # print(row)
        dp = dict()
        dp['index'] = i
        dp['context'] = literal_eval(row[0])
        dp['target'] = row[1]
        dp['references'] = literal_eval(row[2])
        # dp['domain'] = literal_eval(row[3])
        # if dp['domain'][0] == 'out_of_domain':
        #     continue
        dp['nofudge'] = literal_eval(row[3])
        dp['fudge'] = literal_eval(row[4])
        if len(row)>5:
            dp['paths'] = literal_eval(row[5])
        ret.append(dp)

    return ret

def get_specifc_response_types(data, response_key):
    return_data = {'target': [], 'context': [], 'response': [], 'references': []}
    for i, dp in enumerate(data):
        for r in dp[response_key]:
            return_data['target'].append(dp['target'])
            return_data['context'].append(dp['context'])
            return_data['response'].append(r)
            if 'references' in dp:
                return_data['references'].append(dp['references'])

    return return_data

def get_nltksentence_bleu(refs, sys):
    scores = []
    for r, c in zip(refs, sys):
        # tokenized_reference = [x.split() for x in r]
        # tokenized_candidate = c.split()
        tokenized_reference = [regexp_tokenize(x, pattern=pattern) for x in r]
        tokenized_candidate = regexp_tokenize(c, pattern=pattern)
        # print(tokenized_reference, tokenized_candidate)
        score = sentence_bleu(tokenized_reference, tokenized_candidate,smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
        scores.append(score)

    return sum(scores)/len(scores)

def get_sacresentence_bleu(refs, sys):
    scores = []
    for r, c in zip(refs, sys):
        # print(tokenized_reference, tokenized_candidate)
        score = sacrebleu.sentence_bleu(c, r,  smooth_method='floor').score
        scores.append(score)

    return sum(scores)/len(scores)


def print_bleu(data):
    refs = data['references']
    sys = data['response']
    print(len(refs), len(sys))

    bleu = BLEU()
    max_refs = max([len(x) for x in refs])
    refs_permuted = [[] for x in range(max_refs)]
    #refs_permuted will have number of rows equal to max refs, len of row will be equal to num datapounts
    for i,x in enumerate(refs):
        for m in range(max_refs):
            if m<len(refs[i]):
                dp = refs[i][m]
            else:
                dp = ''
            refs_permuted[m].append(dp)
    # import pdb;pdb.set_trace()
    bscore = bleu.corpus_score(sys, refs_permuted)
    print('sacre corpus_score', bscore, bleu.get_signature())
    sacrescore = get_sacresentence_bleu(refs, sys)
    print('sacre sentence_bleu score', sacrescore)
    # print(sys)
    # print(refs)
    # bscore = sacrebleu.corpus_bleu(sys, refs_permuted,  smooth_method='floor')
    # print(bscore, 'sacre corpus_bleu floor smoothing')
    # bscore = sacrebleu.corpus_bleu(sys, refs_permuted,  smooth_method='exp')
    # print(bscore, 'sacre corpus_bleu exp smoothing')
    nltkscore = get_nltksentence_bleu(refs, sys)
    print('nltkscore', nltkscore)

def save_scores(data, out_file, out_file_best, bert_model_args_coherence, bert_model_args_targetcoherence):
    response_keys = ['fudge', 'nofudge', 'references']
    all_res = []
    outfile = open(out_file, 'w')
    outfilebest = open(out_file_best, 'w')
    for i, dp in enumerate(tqdm(data)):
        dp['domain'] = 'in_domain'
        # import pdb;pdb.set_trace()
        for response_key in response_keys:
            examples_tc = []
            for r in dp[response_key]:
                target, context, response = dp['target'], dp['context'], r
                references = dp['references']
                example = {'context': context, 'target': target, 'response': response, 'references':references, 'domain':dp['domain']}
                examples_tc.append(example)
            tc_scores = predict_testpoint_targetcoherence(bert_model_args_targetcoherence, examples_tc)
            max_tc_scores = max(tc_scores)
            max_index = tc_scores.index(max_tc_scores)

            target, context, response, references = dp['target'], dp['context'], dp[response_key][max_index], dp['references']
            example = {'context': context, 'target': target, 'response': response, 'references': references,
                       'domain': dp['domain'], 'path':dp['paths'][max_index]}
            max_tc_scores = tc_scores[max_index]
            example[response_key + '_tc_scores'] = np.float64(max_tc_scores)
            if response_key == 'references':
                nonoverlap_references = [x for x in references if x != response]
            else:
                nonoverlap_references = references
            if len(nonoverlap_references) > 0:
                example[response_key + '_bleu_scores'] = (
                    sacrebleu.sentence_bleu(response, nonoverlap_references, smooth_method='floor').score)
                # info = bertscore_multi_refs_working(
                #     bert_scorer,
                #     bertscore_cands=[response],
                #     bertscore_refs=[nonoverlap_references],
                # )
                # for k,v in info.items():
                #     example[response_key + k] = v
            example['response'] = response
            example['target'] = target
            if 'path' in dp:
                example['path'] = dp['paths'][max_index]
            # import pdb;pdb.set_trace()
            json.dump(example, outfilebest)
            outfilebest.write("\n")
            outfilebest.flush()


            for i, r in enumerate(dp[response_key]):
                target, context, response = dp['target'], dp['context'], r
                references = dp['references']
                # example = {'context': context, 'target': 'State fact :' + target, 'response': response + ' ' + target, 'references':references, 'domain':dp['domain']}
                # print(example)
                # c_scores = predict_testpoint_coherence(bert_model_args_coherence, [example])
                example = {'context': context, 'target': target, 'response': response, 'references':references, 'domain':dp['domain'], 'path':dp['paths'][i]}
                # tc_scores = predict_testpoint_targetcoherence(bert_model_args_targetcoherence, [example])
                tc_score = tc_scores[i]
                # example[response_key+ '_c_scores'] = np.float64(c_scores[0])
                example[response_key+'_tc_scores'] = np.float64(tc_score)
                if response_key == 'references':
                    nonoverlap_references = [x for x in references if x!=response ]
                else:
                    nonoverlap_references = references
                if len(nonoverlap_references)>0:
                    example[response_key + '_bleu_scores'] = (sacrebleu.sentence_bleu(response, nonoverlap_references, smooth_method='floor').score)
                    # info = bertscore_multi_refs_working(
                    #     bert_scorer,
                    #     bertscore_cands=[response],
                    #     bertscore_refs=[nonoverlap_references],
                    # )
                    # for k, v in info.items():
                    #     example[response_key + k] = v
                example['response'] = response
                example['target'] = target
                if 'path' in dp:
                    example['path'] = dp['paths'][i]
                # if 'reference' in response_key:
                #     print(example[response_key+'_tc_scores'])
                all_res.append(example)
                # import pdb;pdb.set_trace()
                json.dump(example, outfile)
                outfile.write("\n")
                outfile.flush()

    outfile.close()
    outfilebest.close()
    return all_res

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', default='')
    parser.add_argument('-i','--input_file', required=True)
    parser.add_argument("--analyze_scores", default=False, action="store_true",
                       help="Flag to analyze scores")
    #parser.add_argument('--num_multi_response', default=5, type=int)
    args = parser.parse_args()


    if args.analyze_scores:
        analyze_scores(args)
        exit(0)
    print("READING DATA....", args.input_file)
    fname = args.input_file
    data = read_predicted_data(fname)
    print("---> data length: ", len(data))
    # print("data[0]: ", data[0] )
    # data = data[:20]

    nofudge_data = get_specifc_response_types(data, 'nofudge')
    references_data = get_specifc_response_types(data, 'references')

    print('\nFor simple fine-tuned')
    print_bleu(nofudge_data)
    print_coco_scores(nofudge_data)

    print('\nFor references, sanity check')
    # print_bleu(references_data)
    print_coco_scores(references_data)
    # exit(0)

    print("="*33)
    print("EVALUATING....")
    # bert_model_args_coherence = get_model_tokenizer_coherence(model_name_or_path="../data_prep/daily_dialogue_act/targetcoherence/tmp/test-both/")
    # bert_model_args_targetcoherence = get_model_tokenizer_targetcoherence(model_name_or_path="../data_prep/daily_dialogue_act/targetcoherence/tmp/test-t1neg_context_response/") #test-t1neg_context_response #test-t2neg_context_response_neg_shortcircuit
    bert_model_args_coherence = None
    bert_model_args_targetcoherence = get_model_tokenizer_targetcoherence(model_name_or_path="test-alv2-withneg_response_target_specific/")
    examples = [{'context': 'i enjoy staring up at the sky.', 'response': "I like stargazing outside with my pet.",
                 'target': 'i like to spend a lot of my free time with my pet.'}, \
                 {'context': 'i enjoy staring up at the sky.', 'response': "I like stargazing outside.",
                 'target': 'i like to spend a lot of my free time with my pet.'}, \
                 {'context': 'i enjoy staring up at the sky.', 'response': "I like walking with my pet.",
                 'target': 'i like to spend a lot of my free time with my pet.'}, \
                {'context': ["excuse me , ma'am . can you tell me where the nearest postoffice is ?", "of course . go straight ahead . turn right at the next street .\
          you 'll see a tall , yellow building . the post office is on the first floor .",
                             'do you mean that i go that way for one block , then turn right ?',
                             'yes , you are right .', 'is it far ?'],
                 'response': "no , it 's only about five minutes'walk .",
                 'target': "no 's it only about five minutes'walk"}]

    # all_scores = predict_testpoint_coherence(bert_model_args_coherence, examples)
    # print(all_scores)
    all_scores = predict_testpoint_targetcoherence(bert_model_args_targetcoherence, examples)
    print(all_scores)

    out_file = '.'.join(args.input_file.split('.')[:-1]) + 'nc_scores.jsonl'
    out_file_best = '.'.join(args.input_file.split('.')[:-1]) + 'nc_bestscores.jsonl'

    save_scores(data, out_file, out_file_best, bert_model_args_coherence, bert_model_args_targetcoherence)

    args.input_file = out_file
    analyze_scores(args)

    print('\nAnalysing with best scored responses')
    args.input_file = out_file_best
    analyze_scores(args)
    ###############

