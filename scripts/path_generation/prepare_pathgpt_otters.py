#!/usr/bin/env python
# coding: utf-8

# In[87]:


# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
import json
import random
import copy
import math
import csv
from tqdm import tqdm
from pprint import pprint
from IPython.display import JSON, display
import random
from operator import itemgetter
# import spacy
import string
import jsonlines


# import yake
# kw_extractor = yake.KeywordExtractor(top=5, dedupLim=0.9)

# nlp = spacy.load("en_core_web_lg")
# STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

def write_json_lines(output_file_name, list_data, output_folder='./'):
    with jsonlines.open(output_folder + output_file_name, mode='w') as writer:
        for dataline in list_data:
            writer.write(dataline)


def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def get_csv_lines(filename):
    lines = []
    with open(filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            lines.append(row)

    return lines


import math

# prob_arr = [[0.3, 0.4, -1.3]]
# [-sum(math.log(x) for x in l if x < 1 and x >= 0) for l in prob_arr]
# In[120]:


# INPUT_FILE = 'learning-generator/ottersout/aug_in_domain_dev_outv1.jsonl'
# INPUT_FILE = 'learning-generator/ottersout/aug_in_domain_train_outv1.jsonl'
# INPUT_FILE = 'learning-generator/ottersout/augfp4_in_domain_train_out.jsonl'
# INPUT_FILE = 'learning-generator/ottersout/augfp4_in_domain_test_out.csv.new.removeoverlap.splitentities.jsonl'
# out_folder = '../naacl-2021-fudge-controlled-generation/ottersdata/cs_paths_pkrv7_expkey3_thres2.0/'

#### EXP1:
# INPUT_FILE = 'learning-generator/ottersout/augfp4_in_domain_test_out.csv.removeoverlap.splitentities.jsonl'
# out_folder = '../naacl-2021-fudge-controlled-generation/ottersdata/cs_paths_fpv4_exp5/'


apply_reranking = True
# apply_ranking_topk = 3


# display(JSON(data_otters[6]))

# In[119]:


rel2text_path = './learning-generator/utils/relation2text.json'
with open(rel2text_path, 'r') as fr:
    relation2text = json.load(fr)
    relation2text = {k.lower(): v for k, v in relation2text.items()}
relation2textset = set(relation2text.keys())
print(relation2textset)

# In[116]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[91]:


#####
import numpy as np


if apply_reranking:
    # wget using https://raw.githubusercontent.com/ScriptSmith/topwords/master/counts.txt
    gutenberg_counts = open('counts.txt', 'r').readlines()
    gutenberg_counts = [s.strip().split() for s in gutenberg_counts]
    gutenberg_word2cnt = {w: int(c) for c, w in gutenberg_counts}
    gutenberg_idf = {w: (1.0 / math.log(1 + c)) for w, c in
                     gutenberg_word2cnt.items()}  # more frequnt words have low frequency
    # more freuqnt words will have low scores in gutenberg_idf
    default_idf_val = 1.0 / (math.log(1 + 1))  # assuming every word occurs at least once


def _idf_score_single(x):
    return gutenberg_idf.get(x, default_idf_val)


def _idf_score(x, y):
    x_score = np.mean([_idf_score_single(xi) for xi in x.strip().split()])
    y_score = np.mean([_idf_score_single(yi) for yi in y.strip().split()])
    return x_score + y_score


if apply_reranking:

    ### testing

    print(gutenberg_idf.get('the', -1))  # # expecting this to be lowest
    print(gutenberg_idf.get('table', -1))
    print(gutenberg_idf.get('solar', -1))
    print(gutenberg_idf.get('logarithm', -1))  # # expecting this to be highest

    print()
    print(_idf_score('table', 'book'))  # expecting this to be highest
    print(_idf_score('give away', 'book'))
    print(_idf_score('give', 'away'))
    print(_idf_score('the', 'is'))  # expecting this to be lowest
    print(_idf_score('is the', 'is'))  # expecting this to be lowest




def convert_edgesnames(path):
    path_words = path.split()
    new_path = []
    for w in path_words:
        new_word = relation2text.get(w, w)
        new_path.append(new_word)
    # print(path, new_path)
    new_path = ' '.join(new_path)
    new_path = new_path.replace('  ', ' ')

    return new_path


def get_path_words(path):
    relations_all = list(relation2textset)
    relations_all.sort()
    for r in relations_all:
        path = path.replace(r, '----')
    path_words = [x.strip() for x in path.split('----')]
    return set(path_words)


def get_filtered_paths(paths, scores, parse_edges=True, input_entities=None, type=''):
    # scores, paths = zip(*sorted(zip(scores, paths))) -- removed as per PG
    min_score, max_scores = min(scores), max(scores)
    list_pathsadded = []
    filt_paths, filt_scores = [], []
    # rel_list = ['atlocation', 'capableof', 'causes', 'causesdesire', 'createdby', 'definedas', 'desireof', 'desires', 'hasa', 'hasfirstsubevent', 'haslastsubevent', 'haspaincharacter', 'haspainintensity', 'hasprerequisite', 'hasproperty', 'hassubevent', 'inheritsfrom', 'instanceof', 'isa', 'locatednear', 'locationofaction', 'madeof', 'motivatedbygoal', 'notcapableof', 'notdesires', 'nothasa', 'nothasproperty', 'notisa', 'notmadeof', 'partof', 'receivesaction', 'relatedto', 'symbolof', 'usedfor']
    # und_rellist = ['_'+x for x in rel_list]
    # relation_set = set(rel_list).union(und_rellist)
    for i, path in enumerate(paths):
        path = path.replace('_ ', '_')
        pathwords = set(path.split())
        pathwords_norel = pathwords - relation2textset
        pathwords_compound = get_path_words(path)
        #         pathwords_norel = get_path_words(path)

        ##if too less entities, rmove - TODO its dangerous might have to remove or write a better version
        if type is not 'aug' and input_entities is not None and (
                len(pathwords_norel) < len(input_entities) - 3):  # or len(pathwords_norel)>len(input_entities)+1):
            # print(pathwords_norel, path, input_entities)
            continue
        if len(pathwords_norel) < 2:
            # print(path, input_entities)
            # make the final part of path same as the enitity itself if the path only has one entity
            # print(path, pathwords_norel)
            path = ', '.join(str(e) for e in pathwords_norel)
            # continue

        # only add paths if ppl<2*min ppl
        if scores[i] < 2 * min_score:
            # print(path, '1--1', filt_paths, path in filt_paths)
            # print(input_entities)
            if input_entities is not None:
                path_specificwords = set(path.split())
                path_specificwords = path_specificwords - relation2textset
                for multiword in input_entities:
                    for word in multiword.split():
                        if word not in input_entities:
                            input_entities.append(word)
                extraents = path_specificwords - set(input_entities)
                # some entities in path are extra
                if len(extraents) > 1:
                    # print(extraents, path, set(input_entities))
                    continue
            if parse_edges:
                path = convert_edgesnames(path)
            if path in filt_paths:  # avooid repetition
                continue
            # print(path, '--', filt_paths, path in filt_paths)
            if pathwords_compound in list_pathsadded:
                continue
            filt_paths.append(path)
            filt_scores.append(scores[i])
            list_pathsadded.append(pathwords_compound)

    return filt_paths, filt_scores


def get_min_path(paths, scores, parse_edges=True):
    min_score, max_scores = min(scores), max(scores)
    filt_paths, filt_scores = [], []
    for i, path in enumerate(paths):
        path = path.replace('_ ', '_')
        if scores[i] == min_score:
            filt_path = path
            filt_score = (scores[i])

    if parse_edges:
        filt_path = convert_edgesnames(filt_path)
    # print(paths, filt_path)
    return filt_path, filt_score


# relations = {"AtLocation":"at location","CapableOf":"capable of","Causes":"causes","CausesDesire":"causes desire","CreatedBy":"created by","DefinedAs":"defined as","DesireOf":"desire of","Desires":"desires","HasA":"has a","HasFirstSubevent":"has first subevent","HasLastSubevent":"has last subevent","HasPainCharacter":"has pain character","HasPainIntensity":"has pain intensity","HasPrerequisite":"has prequisite","HasProperty":"has property","HasSubevent":"has subevent","InheritsFrom":"inherits from","InstanceOf":"instance of","IsA":"is a","LocatedNear":"located near","LocationOfAction":"location of action","MadeOf":"made of","MotivatedByGoal":"motivated by goal","NotCapableOf":"not capable of","NotDesires":"not desires","NotHasA":"not has a","NotHasProperty":"not has property","NotIsA":"not is a","NotMadeOf":"not made of","PartOf":"part of","ReceivesAction":"receives action","RelatedTo":"related to","SymbolOf":"symbol of","UsedFor":"used for"}
# relationsnames = ([x.lower() for x in relations.keys()]); print(relationsnames)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text


extra_stopwords = ["lot", "person", "have", "not", "also", "very", "often", "however", "too", "usually", "really",
                   "early", "never", "always", "sometimes", "together", "likely", "simply", "generally", "instead",
                   "actually", "again", "rather", "almost", "especially", "ever", "quickly", "probably", "already",
                   "below", "directly", "therefore", "else", "thus", "easily", "eventually", "exactly", "certainly",
                   "normally", "currently", "extremely", "finally", "constantly", "properly", "soon", "specifically",
                   "ahead", "daily", "highly", "immediately", "relatively", "slowly", "fairly", "primarily",
                   "completely", "ultimately", "widely", "recently", "seriously", "frequently", "fully", "mostly",
                   "naturally", "nearly", "occasionally", "carefully", "clearly", "essentially", "possibly", "slightly",
                   "somewhat", "equally", "greatly", "necessarily", "personally", "rarely", "regularly", "similarly",
                   "basically", "closely", "effectively", "initially", "literally", "mainly", "merely", "gently",
                   "hopefully", "originally", "roughly", "significantly", "totally", "twice", "elsewhere", "everywhere",
                   "obviously", "perfectly", "physically", "successfully", "suddenly", "truly", "virtually",
                   "altogether", "anyway", "automatically", "deeply", "definitely", "deliberately", "hardly", "readily",
                   "terribly", "unfortunately", "forth", "briefly", "moreover", "strongly", "honestly", "previously",
                   "as", "there", "when", "how", "so", "up", "out", "no", "only", "well", "then", "first", "where",
                   "why", "now", "around", "once", "down", "off", "here", "away", "today", "far", "quite", "later",
                   "above", "yet", "maybe", "otherwise", "near", "forward", "somewhere", "anywhere", "please",
                   "forever", "somehow", "absolutely", "abroad", "yeah", "nowhere", "the", "to", "in", "on", "by",
                   "more", "about", "such", "through", "new", "just", "any", "each", "much", "before", "between",
                   "free", "right", "best", "since", "both", "sure", "without", "back", "better", "enough", "lot",
                   "small", "though", "less", "little", "under", "next", "hard", "real", "left", "least", "short",
                   "last", "within", "along", "lower", "TRUE", "bad", "across", "clear", "easy", "full", "close",
                   "late", "proper", "fast", "wide", "item", "wrong", "ago", "behind", "quick", "straight", "direct",
                   "extra", "pretty", "overall", "alone", "bright", "flat", "whatever", "slow", "clean", "fresh",
                   "whenever", "cheap", "thin", "cool", "fair", "fine", "smooth", "FALSE", "thick", "nearby", "wild",
                   "apart", "none", "strange", "aside", "super", "ill", "honest", "ok", "thanks"]


def process_data(data, INPUT_FILE, out_folder, type_use='direct', parse_edges=True, is_test=False, apply_ranking_topk=3):
    outfile_name = out_folder + INPUT_FILE.split('/')[-1].split('.')[0] + '_' + type_use + '.jsonl'
    if is_test == True: outfile_name = out_folder + INPUT_FILE.split('/')[-1].split('.')[
        0] + '_istest_' + type_use + '.jsonl'
    print(outfile_name, len(data))
    context_target_refs = {}
    for i, dp in enumerate(data[:]):
        dp['context'], dp['target'] = dp['context'].strip(), dp['target'].strip()
        if (dp['context'], dp['target']) not in context_target_refs:
            #             print(dp['context'], dp['target'])
            context_target_refs[(dp['context'], dp['target'])] = []
        context_target_refs[(dp['context'], dp['target'])].append(dp['response'])
    print(len(context_target_refs))
    #     print(context_target_refs.keys())
    with open(outfile_name, 'w') as out:
        for i, dp in enumerate(data[:]):
            all_head_tails = []
            for ht in dp['paths'].keys():
                head, tail = ht.split('---')
                all_head_tails.append((head, tail))

            # TODO add reranking and filter logic
            filtered_head_tails = []
            for ht_pair in all_head_tails:
                h, t = ht_pair
                if h in extra_stopwords or t in extra_stopwords:
                    continue
                if type(h) is str and len(h) < 2 or len(t) < 2:
                    continue
                filtered_head_tails.append(ht_pair)
            reranked_head_tails = filtered_head_tails

            ##add reranking
            if apply_reranking:
                all_head_tails_scores = [[x, y, _idf_score(x, y)] for x, y in reranked_head_tails]
                all_head_tails_scores = sorted(all_head_tails_scores, key=lambda k: -k[2])  # decreasing score
                reranked_head_tails = [[x, y] for x, y, _ in all_head_tails_scores[:apply_ranking_topk]]

                # reranked_head_tails = reranked_head_tails

            for ht in reranked_head_tails:
                ht = ht[0] + '---' + ht[1]
                head, tail = ht.split('---')
                if 'direct' in type_use:
                    paths, scores = dp['paths'][ht]['headtotail_paths'], dp['paths'][ht]['headtotail_scores']

                    if is_test is True:
                        path, score = get_min_path(paths, scores, parse_edges=True)
                        newdp = {'context': dp['context'], 'target': dp['target'], 'response': dp['response']}
                        newdp['path'] = path
                        newdp['score_path'] = score
                        newdp['type'] = 'direct'
                        newdp['path_headentity'] = head
                        newdp['path_tailentity'] = tail
                        newdp['outputs'] = context_target_refs[(dp['context'], dp['target'])]
                        json.dump(newdp, out)
                        out.write('\n')
                        continue

                    paths, scores = get_filtered_paths(paths, scores, parse_edges=True)
                    for i, path in enumerate(paths):
                        # newdp = copy.deepcopy(dp)
                        newdp = {'context': dp['context'], 'target': dp['target'], 'response': dp['response']}
                        newdp['path'] = path
                        newdp['score_path'] = scores[i]
                        newdp['type'] = 'direct'
                        newdp['path_headentity'] = head
                        newdp['path_tailentity'] = tail
                        newdp['outputs'] = context_target_refs[(dp['context'], dp['target'])]
                        json.dump(newdp, out)
                        out.write('\n')
                if 'augkeywords' in type_use:
                    withentity_val = dp['paths'][ht]['augkeywords_intermediate']
                    paths, scores = withentity_val['withentity_paths'], withentity_val['withentity_scores']
                    entities_inpath = withentity_val['entities_inpath']
                    input_entities = [head, tail] + dp['context_keywords'] + dp['target_keywords']
                    for x in entities_inpath:
                        input_entities += x.split()
                    # print(input_entities)
                    if is_test is True:
                        path, score = get_min_path(paths, scores, parse_edges=True)
                        newdp = {'context': dp['context'], 'target': dp['target'], 'response': dp['response']}
                        newdp['path'] = path
                        newdp['score_path'] = score
                        newdp['type'] = 'augkeywords'
                        newdp['path_headentity'] = head
                        newdp['path_tailentity'] = tail
                        newdp['entities_inpath'] = entities_inpath
                        newdp['outputs'] = context_target_refs[(dp['context'], dp['target'])]
                        json.dump(newdp, out)
                        out.write('\n')
                    else:
                        paths, scores = get_filtered_paths(paths, scores, parse_edges=True,
                                                           input_entities=input_entities, type='aug')
                        for i, path in enumerate(paths):
                            if head not in path:
                                continue
                            newdp = {'context': dp['context'], 'target': dp['target'], 'response': dp['response']}
                            newdp['path'] = path
                            newdp['score_path'] = scores[i]
                            newdp['type'] = 'augkeywords'
                            newdp['path_headentity'] = head
                            newdp['path_tailentity'] = tail
                            newdp['entities_inpath'] = entities_inpath
                            newdp['outputs'] = context_target_refs[(dp['context'], dp['target'])]
                            json.dump(newdp, out)
                            out.write('\n')

                if 'goldkeywords' in type_use:
                    withentity_val = dp['paths'][ht]['goldkeywords_intermediate']
                    paths, scores = withentity_val['withentity_paths'], withentity_val['withentity_scores']
                    entities_inpath = withentity_val['entities_inpath']
                    input_entities = [head, tail]
                    for x in entities_inpath:
                        input_entities += x.split()
                    if is_test is True:
                        path, score = get_min_path(paths, scores, parse_edges=True)
                        newdp = {'context': dp['context'], 'target': dp['target'], 'response': dp['response']}
                        newdp['path'] = path
                        newdp['score_path'] = score
                        newdp['type'] = 'goldkeywords'
                        newdp['path_headentity'] = head
                        newdp['path_tailentity'] = tail
                        newdp['entities_inpath'] = entities_inpath
                        newdp['outputs'] = context_target_refs[(dp['context'], dp['target'])]
                        json.dump(newdp, out)
                        out.write('\n')
                    else:
                        paths, scores = get_filtered_paths(paths, scores, parse_edges=True,
                                                           input_entities=input_entities)
                        for i, path in enumerate(paths):
                            if head not in path:
                                continue
                            newdp = {'context': dp['context'], 'target': dp['target'], 'response': dp['response']}
                            newdp['path'] = path
                            newdp['score_path'] = scores[i]
                            newdp['type'] = 'goldkeywords'
                            newdp['path_headentity'] = head
                            newdp['path_tailentity'] = tail
                            newdp['entities_inpath'] = entities_inpath
                            newdp['outputs'] = context_target_refs[(dp['context'], dp['target'])]
                            json.dump(newdp, out)
                            out.write('\n')


#                     if 'splitentitypath' in type_use:
#                         headtoentity_paths, headtoentity_scores = withentity_val['headtoentity_paths'], withentity_val['headtoentity_scores']
#                         headtoentity_paths, headtoentity_scores = get_filtered_paths(headtoentity_paths, headtoentity_scores, parse_edges=True)
#                         entitytotail_paths, entitytotail_scores = withentity_val['entityto_paths'], withentity_val['entityto_scores']
#                         entitytotail_paths, entitytotail_scores = get_filtered_paths(entitytotail_paths, entitytotail_scores, parse_edges=True)
#                         for indh, htoe in enumerate(headtoentity_paths):
#                             for indt, etot in enumerate(entitytotail_paths):
#                                 # if head not in htoe or tail not in etot:
#                                 #     print('ABsent', head, '..', tail, '--', htoe, '-->' ,etot)
#                                 #     continue
#                                 newdp = {'context':dp['context'], 'target': dp['target'], 'response': dp['response']}
#                                 newdp['path'] = htoe + ' ' + remove_prefix(etot, withentity)
#                                 newdp['score_path'] = headtoentity_scores[indh] + entitytotail_scores[indt]
#                                 newdp['type'] = 'splitentitypath'
#                                 newdp['path_headentity'] = head
#                                 newdp['path_tailentity'] = tail
#                                 newdp['withentity'] = withentity
#                                 # newdp['headtoentity_path'] = htoe
#                                 # newdp['entitytotail_path'] = etot
#                                 json.dump(newdp, out)
#                                 out.write('\n')
# for kchild in dp['paths'][k].keys():
#     if '_paths' in kchild:


# out_folder = '../naacl-2021-fudge-controlled-generation/ottersdata/cs_paths_fpv4/'
# #type_use direct augkeywords goldkeywords
# process_data(data, INPUT_FILE, out_folder, type_use='direct', parse_edges=True, is_test=False)
# process_data(data, INPUT_FILE, out_folder, type_use='augkeywords', parse_edges=True, is_test=False)
# process_data(data, INPUT_FILE, out_folder, type_use='goldkeywords', parse_edges=True, is_test=False)


# In[118]:

INPUT_FILES = ['learning-generator/ottersout/augfp4_in_domain_train_out.csv.new.removeoverlap.splitentities.jsonl',
    'learning-generator/ottersout/augfp4_in_domain_dev_out.csv.new.removeoverlap.splitentities.jsonl',
    'learning-generator/ottersout/augfp4_in_domain_test_out.csv.new.removeoverlap.splitentities.jsonl'
               ]
out_folders = ['../naacl-2021-fudge-controlled-generation/ottersdata/cs_paths_expkey1_nozip/',
        '../naacl-2021-fudge-controlled-generation/ottersdata/cs_paths_expkey2_nozip/',
        '../naacl-2021-fudge-controlled-generation/ottersdata/cs_paths_expkey4_nozip/']

import os


for apply_ranking_topk,out_folder in zip([1,2,4],out_folders):

    for INPUT_FILE in INPUT_FILES:

        data_otters = get_json_lines(INPUT_FILE)
        print()
        print(" ---->>>>> len(data_otters) = ", len(data_otters))
        print(" ---->>>>> apply_ranking_topk = ", apply_ranking_topk )
        print(" ---->>>>> out_folder = ", out_folder )
        # os.mkdir(out_folder)
        print()

        # type_use direct augkeywords goldkeywords
        process_data(data_otters[:], INPUT_FILE, out_folder, type_use='direct', parse_edges=True, is_test=False, apply_ranking_topk=apply_ranking_topk)
        process_data(data_otters, INPUT_FILE, out_folder, type_use='augkeywords', parse_edges=True, is_test=False)
        process_data(data_otters, INPUT_FILE, out_folder, type_use='goldkeywords', parse_edges=True, is_test=False, apply_ranking_topk=apply_ranking_topk)

        # type_use direct augkeywords goldkeywords
        process_data(data_otters, INPUT_FILE, out_folder, type_use='direct', parse_edges=True, is_test=True, apply_ranking_topk=apply_ranking_topk)
        process_data(data_otters, INPUT_FILE, out_folder, type_use='augkeywords', parse_edges=True, is_test=True, apply_ranking_topk=apply_ranking_topk)
        process_data(data_otters, INPUT_FILE, out_folder, type_use='goldkeywords', parse_edges=True, is_test=True, apply_ranking_topk=apply_ranking_topk)

    # break
    print(" ===================================== ")

