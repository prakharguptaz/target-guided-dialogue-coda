# import nltk
import itertools
import logging
import json
import csv
import random
import os
import pandas as pd
from tqdm import tqdm
# TYPE_OF_TARGET = 'srlcommand_touse'
# def get_target(batch):
#     if TYPE_OF_TARGET and TYPE_OF_TARGET!='':
#         if TYPE_OF_TARGET not in batch:
#             target = ''
#             print('TYPE_OF_TARGET not present')
#         else:
#             endtarget_options = []
#             target_options = batch[TYPE_OF_TARGET]
#             if target_options and len(target_options)>0:
#                 target_options = target_options[-1]
#                 endtarget_options = [x for x in target_options if x!='']
#             if len(endtarget_options)>0:
#                 target = random.choice(endtarget_options)
#             else:
#                 target = ''

#     return target

def get_target(batch):

    return batch.get('target_clause', '')


def get_json_data(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_dp(ind, context, target, response, labeltype, label, original_target):
    dp = dict()
    dp['index'] = ind
    dp['context'] = context
    dp['target'] = target
    dp['response'] = response #+ ' ' + original_target
    dp['label'] = label
    dp['type'] = labeltype

    return dp


def prepare_data_train(input_file, dest_file, type_split, num_random_negatives = 3):
    data = pd.read_csv(input_file)
    # data = data.head(5)
    print(data.shape)
    # data = data.reset_index(drop=True)
    num_columns = data.shape[1]
    if num_columns==4:
        data.columns = ['index','source', 'target', 'response']
    else:
        data.columns = ['index', 'source', 'target', 'response'] + ['neg_response'+str(i) for i in range(num_columns-4)]
        # data.columns = [0,1,2,3] + [i for i in range(num_columns-4)]
    # import pdb;pdb.set_trace()
    list_responses = data['response'].tolist()
    list_targets = data['target'].tolist()
    list_context = data['source'].tolist()
    target_response_dict = dict()
    context_target_response_dict = dict()
    set_neg_responses = set()
    for ind, dp in enumerate(data.iterrows()):
        row = dp[1]
        context = row['source']
        target = row['target']
        response = row['response']
        if target not in target_response_dict:
            target_response_dict[target] = []
        target_response_dict[target].append(response)

        if context+target not in context_target_response_dict:
            context_target_response_dict[context+target] = []
        context_target_response_dict[context+target].append(response)

    new_data = []
    for ind, dp in enumerate(tqdm(data.iterrows())):
        dp = dp[1]
        context = dp['source']
        target = dp['target']
        positive_response = dp['response']

        # newdp = get_dp(ind, context, target, positive_response, 'positive', 1)
        # new_data.append(newdp)
        # random_dp = random.choice(data)
        
        # for k in dp.keys():
        #     if 'neg_response' in k:
        #         neg_response = dp[k]
        #         newdp = get_dp(ind, context, target, neg_response, 'neg_response_genwithrandomtarget', 0, target)
        #         new_data.append(newdp)
        #         newdp = get_dp(ind, context, target, positive_response, 'positive', 1, target)
        #         new_data.append(newdp)
                


        # target_specific_responses_list = target_response_dict[target]
        # target_specific_responses = random.sample(target_specific_responses_list, min(len(target_specific_responses_list),num_random_negatives))
        # for i, nr in enumerate(target_specific_responses):
        #     tries = 30
        #     # if nr=='I love helping people learn':
        #     #     import pdb;pdb.set_trace()
        #     while nr == positive_response or (context+target in context_target_response_dict and nr in context_target_response_dict[context+target]):
        #         tries-=1
        #         nr = random.choice(target_specific_responses_list)
        #         if tries<0:
        #             break
        #     if tries<0:
        #         continue
        #     # negative_responses[i] = nr
        #     if type_split =='train':
        #         newdp = get_dp(ind, context, target, nr, 'neg_response_target_specific', 0, target)
        #         if context+target+nr in set_neg_responses:
        #             continue
        #         new_data.append(newdp)
        #         set_neg_responses.add(context+target+nr)
        #         newdp = get_dp(ind, context, target, positive_response, 'positive', 1, target)
        #         new_data.append(newdp)

                # print(ind, context, 'negresponse:', nr, 'tgt:', target)
                # import pdb;pdb.set_trace()


        #use target sentence as response
        # if type_split =='train':
        #     newdp = get_dp(ind, context, target, target, 'neg_response_target', 0, target)
        #     new_data.append(newdp)
        #     newdp = get_dp(ind, context, target, positive_response, 'positive', 1, target)
        #     new_data.append(newdp)


        negative_responses = random.sample(list_responses, min(len(list_responses), num_random_negatives))
        for i, nr in enumerate(negative_responses):
            while nr == positive_response or (context+target in context_target_response_dict and nr in context_target_response_dict[context+target]):
                nr = random.choice(list_responses)
            negative_responses[i] = nr
            newdp = get_dp(ind, context, target, nr, 'neg_response_random', 0, target)
            new_data.append(newdp)
            newdp = get_dp(ind, context, target, positive_response, 'positive', 1, target)
            new_data.append(newdp)

        negative_contexts = random.sample(list_context, min(len(list_context), num_random_negatives))
        for i, nc in enumerate(negative_contexts):
            while nc == context:
                nc = random.choice(list_context)
            # negative_contexts[i] = nc
            newdp = get_dp(ind, nc, target, positive_response, 'neg_context_random', 0, target)
            new_data.append(newdp)
            newdp = get_dp(ind, context, target, positive_response, 'positive', 1, target)
            new_data.append(newdp)


        random_targets =  random.sample(list_targets, min(len(list_targets), num_random_negatives))
        for i, random_target in enumerate(random_targets):
            while random_target==target:
                random_target = random.choice(list_targets)
            newdp = get_dp(ind, context, random_target, positive_response, 'neg_target_random', 0, random_target)
            new_data.append(newdp)
            newdp = get_dp(ind, context, target, positive_response, 'positive', 1, target)
            new_data.append(newdp)

        # import pdb;pdb.set_trace()

    output_file = open(dest_file, 'w', encoding='utf-8')
    # import pdb;pdb.set_trace()

    if type_split !='train':
        new_data = new_data[:10000]

    for dic in new_data:
        json.dump(dic, output_file) 
        output_file.write("\n")        



def main():
    FOLDER_NAME = 'ottersnegdata/'
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    input_file = 'tcdata_wneg/total_train_wneg.csv'
    dest_file = FOLDER_NAME+'total_train_wneg.jsonl'
    prepare_data_train(input_file, dest_file, 'train')
    input_file = 'tcdata_wneg/total_dev_wneg.csv'
    dest_file = FOLDER_NAME+'total_dev_wneg.jsonl' #'ottersnegdata/total_dev_negresp.jsonl'
    prepare_data_train(input_file, dest_file, 'dev')
    input_file = 'tcdata_wneg/total_test_wneg.csv'
    dest_file = FOLDER_NAME+'total_test_wneg.jsonl' #'ottersnegdata/total_test_negresp.jsonl'
    prepare_data_train(input_file, dest_file, 'test')

if __name__ == '__main__':
    main()




