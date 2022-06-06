import os
import pickle
import torch
import json
from collections import defaultdict, OrderedDict
import random
from tqdm import tqdm, trange
class PreprocessData(object):
    """docstring for PreprocessData"""
    def __init__(self, data_dir, tokenizer=None):
        super(PreprocessData, self).__init__()
        train_path = os.path.join(data_dir, 'train.txt')
        dev_path = os.path.join(data_dir, 'dev.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        rel2text_path = os.path.join('./utils/', 'relation2text.json')
        relation_vocab_path = os.path.join(data_dir, 'relation_vocab.pkl')
        self.token_path = os.path.join(data_dir, 'token_gpt.pkl')

        with open(rel2text_path, 'r') as fr:
            self.relation2text = json.load(fr)

        self.tokenizer = tokenizer

        self.tokenizer.add_tokens(['<PAD>'])
        self.tokenizer.add_tokens(['<SEP>'])
        self.tokenizer.add_tokens(['<END>'])
        self.tokenizer.add_tokens(['<contains>'])
        self.tokenizer.add_tokens(['<final>'])
        self.PAD = self.tokenizer.convert_tokens_to_ids('<PAD>')
        self.SEP = self.tokenizer.convert_tokens_to_ids('<SEP>')
        self.END = self.tokenizer.convert_tokens_to_ids('<END>')
        self.CONTAINS = self.tokenizer.convert_tokens_to_ids('<contains>')
        self.FINAL = self.tokenizer.convert_tokens_to_ids('<final>')
        # self.load_relation_vocab(relation_vocab_path)

        self.relationsfound = set()

        relation_keys = self.relation2text.keys()
        self.relation_token_list = [x.lower() for x in relation_keys] #+ ['_' + x.lower() for x in relation_keys]
        self.tokenizer.add_tokens(self.relation_token_list)
        print(self.relation_token_list)

        if not os.path.exists(self.token_path):

            token_dataset = {}
            token_dataset['train'] = self.text2token_cnpt(train_path)
            token_dataset['dev'] = self.text2token_cnpt(dev_path)
            token_dataset['test'] = self.text2token_cnpt(test_path)

            with open(self.token_path, 'wb') as handle:
                pickle.dump(token_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_relation2text(self, data_path):
        with open(data_path, 'r') as fr:
            self.relation2text = json.load(fr)
        rel2id = {}
        id2rel = []
        for rel in self.relation2text:
            rel2id[rel] = len(id2rel)
            id2rel.append(rel)
        self.num_relation = len(id2rel)
        self.r2i = rel2id
        self.i2r = id2rel

    def load_relation_vocab(self, data_path):
        with open(data_path, 'rb') as fr:
            rel_vocab = pickle.load(fr)
        self.r2i = rel_vocab['r2i']
        self.i2r = rel_vocab['i2r']
        self.num_relation = len(self.i2r)
        for rel in self.i2r:
            # self.tokenizer.add_tokens(['<' + rel + '>'])
            rel = '<' + rel.replace('<', '').replace('>', '') + '>'
            self.tokenizer.add_tokens([rel])

    def text2token_cnpt(self, data_path):
        input_list = []
        # cnt_line = 0
        max_context_len = 32
        max_label_len = 64
        with open(data_path, 'r') as fr:
            for i, line in enumerate(tqdm(fr, desc=data_path)):
                # if i>5000:
                #     exit(0)
                line_split = line.strip().split('\t')
                current_idx = 0
                text = ''
                intermediate_ents = []
                for _idx, element in enumerate(line_split[1:]):
                    # import pdb;pdb.set_trace()
                    if _idx % 2 != 0:
                        ent_words = element.replace('_', ' ')
                        text += ent_words
                        intermediate_ents+=[ent_words]
                    else:
                        text += ' ' + element + ' '

                for to in text.split():
                    if '_' in to:
                        self.relationsfound.add(to)

                _input = self.tokenizer.encode(text)[:max_label_len]
                # import pdb;                pdb.set_trace()
                _input += [self.PAD] * (max_label_len - len(_input))
                # context = line_split[-1].replace('_', ' ') + '<SEP>' + line_split[0].replace('_', ' ')
                # print(len(intermediate_ents[:-1]))
                if len(intermediate_ents[:-1])>0 and random.random()>0.5:
                    # chosen_ent = random.choice(intermediate_ents[:-1])
                    # context = '<contains>'+ chosen_ent + '<final>' +line_split[-1].replace('_', ' ') + '<SEP>' + line_split[0].replace('_', ' ')
                    mixed_ents = intermediate_ents[:-1]
                    random.shuffle(mixed_ents)
                    context = '<contains>' + '<contains>'.join(mixed_ents) + '<final>' +line_split[-1].replace('_', ' ') + '<SEP>' + line_split[0].replace('_', ' ')
                    # print(intermediate_ents[:-1])
                    # print(text, '---', context)
                    context = self.tokenizer.encode(context)[:max_context_len]
                    context += [self.PAD] * (max_context_len - len(context))
                    _inputcopy = (context + _input[:] + [self.END])
                    # print('xxx', self.tokenizer.decode(_inputcopy))
                    input_list.append(_inputcopy)

                context = line_split[-1].replace('_', ' ') + '<SEP>' + line_split[0].replace('_', ' ')
                # print(intermediate_ents[:-1])
                # print(text, '---', context)
                context = self.tokenizer.encode(context)[:max_context_len]
                context += [self.PAD] * (max_context_len - len(context))
                _input = (context + _input + [self.END])
                # print(self.tokenizer.decode(_input))
                input_list.append(_input)

        return input_list

