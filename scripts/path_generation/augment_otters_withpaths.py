from tqdm import tqdm
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
import string
import yake
kw_extractor = yake.KeywordExtractor(top=5, dedupLim=0.9)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformers
assert transformers.__version__ == '2.8.0'
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
import torch.nn.functional as F
# from typing import bool

import os
import pickle
# rel_path = os.path.join('learning-generator/data/conceptnet/', 'relation_vocab.pkl')
# # ent_path = os.path.join(data_dir, 'entity_vocab.pkl')
# with open(rel_path, 'rb') as handle:
#     rel_vocab = pickle.load(handle)

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import spacy
nlp = spacy.load("en_core_web_sm")

PATH_COMMONSENSE_MODEL = '../../../goal_driven_dialogue/Commonsense-Path-Generator/learning-generator'

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


stopwords = nlp.Defaults.stop_words

from IPython.display import display

lemmatizer = nltk.WordNetLemmatizer()

# Rule for NP chunk and VB Chunk
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        {<RB.?>*<VB.?>*<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...

"""
grammarnoun = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}

"""

grammarverb = r"""
    NBAR:
        {<RB.?>*<VB.?>*<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases

    NP:
        {<NBAR>}

"""
# Chunking
# cp = nltk.RegexpParser(grammar)
cpnoun = nltk.RegexpParser(grammarnoun)

cpverb = nltk.RegexpParser(grammarverb)




def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()

def check_overlap(head_entity, tail_entity) -> bool:
    w = set(head_entity.strip().split())
    t = set(tail_entity.strip().split())
    if len(w)==0 or len(t)==0:
        return False
    # print("w = ", w, " || t = ", t)
    inter = w.intersection(t)
    if len(inter) > 0:
        # print(" -- True")
        return True
    # print(" -- False")
    return False
skip_overlap_cnt = 0
augmentation_cnt = 0

def get_word_postag(word):
    if pos_tag([word])[0][1].startswith('J'):
        return wordnet.ADJ
    if pos_tag([word])[0][1].startswith('V'):
        return wordnet.VERB
    if pos_tag([word])[0][1].startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN


def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    postag = get_word_postag(word)
    word = lemmatizer.lemmatize(word, postag)
    return word


def get_terms(tree):
    for leaf in leaves(tree):
        terms = [normalise(w) for w, t in leaf]
        yield terms


def get_features(document):
    # word tokenizeing and part-of-speech tagger
    tokens = [nltk.word_tokenize(sent) for sent in [document]]
    postag = [nltk.pos_tag(sent) for sent in tokens][0]

    # the result is a tree
    tree = cpnoun.parse(postag)

    terms = get_terms(tree)

    features = []
    for term in terms:
        _term = ''
        for word in term:
            _term += ' ' + word
        features.append(_term.strip())

    tree = cpverb.parse(postag)

    terms = get_terms(tree)

    features_verb = []
    for term in terms:
        _term = ''
        for word in term:
            _term += ' ' + word
        features_verb.append(_term.strip())

    return features, features_verb


def get_verbnouns_simple(document):
    extra_stopwords = ['lot', 'person']
    document = document.lower()
    if document.startswith('i '):
        document = document.replace("i ", 'person ')
    document = document.replace(" i ", ' person ')
    document = document.replace(" he ", ' person ')
    document = document.replace(" she ", ' person ')
    document = document.replace(" they ", ' people ')
    document = document.replace("don't", 'do not')
    document = document.replace("can't", 'can not')
    document = document.replace("won't", 'would not')
    features, features_verb = get_features(document)
    # print(features, features_verb)
    allstopwords = stopwords.union(extra_stopwords)
    res = [x for x in features + features_verb if x not in allstopwords]
    res = [x.replace('person ', '') if 'person ' in x else x for x in res]
    return res


def get_verbnouns(document):
    extra_stopwords = ["lot", "person", "have", "not", "also", "very", "often", "however", "too", "usually", "really", "early", "never", "always", "sometimes", "together", "likely", "simply", "generally", "instead", "actually", "again", "rather", "almost", "especially", "ever", "quickly", "probably", "already", "below", "directly", "therefore", "else", "thus", "easily", "eventually", "exactly", "certainly", "normally", "currently", "extremely", "finally", "constantly", "properly", "soon", "specifically", "ahead", "daily", "highly", "immediately", "relatively", "slowly", "fairly", "primarily", "completely", "ultimately", "widely", "recently", "seriously", "frequently", "fully", "mostly", "naturally", "nearly", "occasionally", "carefully", "clearly", "essentially", "possibly", "slightly", "somewhat", "equally", "greatly", "necessarily", "personally", "rarely", "regularly", "similarly", "basically", "closely", "effectively", "initially", "literally", "mainly", "merely", "gently", "hopefully", "originally", "roughly", "significantly", "totally", "twice", "elsewhere", "everywhere", "obviously", "perfectly", "physically", "successfully", "suddenly", "truly", "virtually", "altogether", "anyway", "automatically", "deeply", "definitely", "deliberately", "hardly", "readily", "terribly", "unfortunately", "forth", "briefly", "moreover", "strongly", "honestly", "previously", "as", "there", "when", "how", "so", "up", "out", "no", "only", "well", "then", "first", "where", "why", "now", "around", "once", "down", "off", "here", "away", "today", "far", "quite", "later", "above", "yet", "maybe", "otherwise", "near", "forward", "somewhere", "anywhere", "please", "forever", "somehow", "absolutely", "abroad", "yeah", "nowhere", "the", "to", "in", "on", "by", "more", "about", "such", "through", "new", "just", "any", "each", "much", "before", "between", "free", "right", "best", "since", "both", "sure", "without", "back", "better", "enough", "lot", "small", "though", "less", "little", "under", "next", "hard", "real", "left", "least", "short", "last", "within", "along", "lower", "TRUE", "bad", "across", "clear", "easy", "full", "close", "late", "proper", "fast", "wide", "item", "wrong", "ago", "behind", "quick", "straight", "direct", "extra", "pretty", "overall", "alone", "bright", "flat", "whatever", "slow", "clean", "fresh", "whenever", "cheap", "thin", "cool", "fair", "fine", "smooth", "FALSE", "thick", "nearby", "wild", "apart", "none", "strange", "aside", "super", "ill", "honest", "ok", "thanks"]
    document = document.lower()
    if document.startswith('i '):
        document = document.replace("i ", 'person ')
    document = document.replace(" i ", ' person ')
    document = document.replace(" he ", ' person ')
    document = document.replace(" she ", ' person ')
    document = document.replace(" they ", ' people ')
    document = document.replace("don't", 'do not')
    document = document.replace("can't", 'can not')
    document = document.replace("won't", 'would not')
    features, features_verb = get_features(document)
    # print(features, features_verb)
    document_words = document.split()
    document_words = [x for x in document_words if
                      x not in ["a", "the", "an", "about", "above", "across", "after", "against", "among", "around",
                                "at", "before", "behind", "below", "beside", "between", "by", "down", "during", "for",
                                "from", "in", "inside", "into", "near", "of", "off", "on", "out", "over", "through",
                                "to", "toward", "under", "up", "with"]]
    noune_indexdict = {}
    for noune in features:
        for i, x in enumerate(document_words):
            if noune in x and len(noune) < len(x) + 3:
                noune_indexdict[noune] = i

    verb_phrases = []
    for verbe in features_verb:
        verbe_index = -1
        for i, x in enumerate(document_words):
            if verbe in x and len(verbe) < len(x) + 3:
                verbe_index = i
        # print(noune_indexdict)
        # verbe_index = document.find(verbe)
        if verbe_index != -1:
            # print(verbe, verbe_index)
            for noune in noune_indexdict:
                # print(verbe, noune, noune_indexdict[noune],verbe_index)
                if noune_indexdict[noune] > verbe_index and noune_indexdict[noune] - verbe_index < 3:
                    # print(verbe, noune)
                    verb_phrases.append(verbe + ' ' + noune)

    features = [x for x in features if x not in ' '.join(verb_phrases)]
    features_verb = [x for x in features_verb if x not in ' '.join(verb_phrases)]
    features_verb = [x.replace('have', '').replace("'ve", '').strip() for x in features_verb]
    allstopwords = stopwords.union(extra_stopwords)
    res = [x for x in features + features_verb + verb_phrases if x not in allstopwords]
    res = [x for x in res if len(x)>2]
    res = [x.replace('person ', '') if 'person ' in x else x for x in res]
    return res


class Generator(nn.Module):
    def __init__(self, gpt, config, max_len=64, temperature=0.7):
        super(Generator, self).__init__()
        self.gpt = gpt
        self.config = config
        self.max_len = max_len
        # self.temperature = temperature
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), filter_tokens=None,
                              min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if filter_tokens is not None:
            for x in filter_tokens:
                if logits.shape[1] > x:
                    logits[:, x] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def forward_greedy(self, inputs):
        # input: [batch, seq]
        context_len = inputs.size(1)
        generated = inputs
        next_token = inputs
        past = None
        with torch.no_grad():
            for step in range(self.max_len):
                outputs = self.gpt(next_token, past=past)
                hidden = outputs[0][:, -1]
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
        """ Generates a sequence of tokens
            Args:
                model: gpt/gpt2 model
                context: tokenized text using gpt/gpt2 tokenizer
                length: length of generated sequence.
                device: torch.device object.
                temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        """
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0)
        generated = context
        with torch.no_grad():
            for _ in tnrange(length):
                inputs = {'input_ids': generated}
                outputs = self.model(
                    **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self, top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def forward(self, inputs, temperature=1, num_outs=1, top_k=0, top_p=0.0):
        # input: [batch, seq]
        if num_outs > 1:
            inputs = inputs.repeat(num_outs, 1)
        context_len = inputs.size(1)
        # print(inputs.shape)
        generated = inputs
        next_token = inputs
        past = None
        probs_arr = [[] for i in range(inputs.shape[0])]
        with torch.no_grad():
            for step in range(self.max_len):
                outputs = self.gpt(next_token, past=past)
                hidden = outputs[0][:, -1]
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_token_logits = next_token_logits / temperature
                # filtered_logits = self.top_k_top_p_filtering(next_token_logits.squeeze(dim=0), top_k=top_k, top_p=top_p)
                filter_tokens = [50268, 50269]
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, filter_tokens=filter_tokens)
                # next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                softmax_outs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(softmax_outs, num_samples=1)
                probs_cur = [softmax_outs[i, x[0]].item() for i, x in enumerate(next_token)]
                for i, x in enumerate(probs_arr):
                    # print(i, probs_cur[i], '--', probs_arr[i])
                    probs_arr[i] += [probs_cur[i]]
                # print(softmax_outs.shape, next_token.shape, next_token, probs_cur)
                # print(probs_arr)
                # next_token = next_token.unsqueeze(dim=0)
                generated = torch.cat((generated, next_token), dim=1)
        # print(probs_arr)
        return generated, probs_arr

lm_type = 'gpt2'
device = torch.device('cuda')
config = GPT2Config.from_pretrained(lm_type)
tokenizer = GPT2Tokenizer.from_pretrained(lm_type)
tokenizer.add_tokens(['<PAD>'])
tokenizer.add_tokens(['<SEP>'])
tokenizer.add_tokens(['<END>'])
tokenizer.add_tokens(['<contains>'])
tokenizer.add_tokens(['<final>'])

#comment below if there is a size mismatch error. there were nt added in first few models
rel2text_path = './relation2text.json'
with open(rel2text_path, 'r') as fr:
    relation2text = json.load(fr)
relation_keys = relation2text.keys()
relation_token_list = [x.lower() for x in relation_keys] #+ ['_' + x.lower() for x in relation_keys]
tokenizer.add_tokens(relation_token_list)



gpt = GPT2Model.from_pretrained(lm_type)
config.vocab_size = len(tokenizer)
gpt.resize_token_embeddings(len(tokenizer))
pretrain_generator_ckpt = PATH_COMMONSENSE_MODEL+"/checkpoints_6lendict_wcontains/model.ckpt"
print('loading generator')
generator = Generator(gpt, config)
generator.load_state_dict(torch.load(pretrain_generator_ckpt, map_location='cpu'))
print('loaded state dict generator')
generator = generator.to(device)


def prepare_input(head_entity, tail_entity, input_len=32):
    head_entity = head_entity.replace('_', ' ')
    tail_entity = tail_entity.replace('_', ' ')
    input_token = tail_entity + '<SEP>' + head_entity
    input_id = tokenizer.encode(input_token, add_special_tokens=False)[:input_len]
    input_id += [tokenizer.convert_tokens_to_ids('<PAD>')] * (input_len - len(input_id))
    return torch.tensor([input_id], dtype=torch.long)


def connect_entities(head_entity, tail_entity, temperature=1, num_outs=1, top_k=0, top_p=1.0):
    gen_input = prepare_input(head_entity, tail_entity)
    gen_input = gen_input.to(device)
    gen_output, prob_arr = generator(gen_input, temperature=temperature, num_outs=num_outs, top_k=top_k, top_p=top_p)
    prob_sum = [-sum(math.log(x) for x in l if x < 1 and x >= 0) for l in prob_arr]
    outs = []
    for gen in gen_output:
        path = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
        path = ' '.join(path.replace('<PAD>', '').split())
        try:
            out = path[path.index('<SEP>') + 6:]
        except:
            print('weird path', path)
            out = path[16:]
        outs.append(out)
    return outs, prob_sum


def prepare_input_wcontains(head_entity, tail_entity, withentities, input_len=32):
    head_entity = head_entity.replace('_', ' ')
    tail_entity = tail_entity.replace('_', ' ')
    if type(withentities) is list:
        withentity = '<contains>'.join(withentities)
    withentity = withentity.replace('_', ' ')
    if len(withentity)>0:
        input_token = '<contains>'+ withentity + '<final>' + tail_entity + '<SEP>' + head_entity
    else:
        input_token = tail_entity + '<SEP>' + head_entity
    input_id = tokenizer.encode(input_token, add_special_tokens=False)
    tries = 0
    while len(input_id)>input_len and tries<10:
        withentities = withentities[1:]
        input_token = '<contains>'+ '<contains>'.join(withentities) + '<final>' + tail_entity + '<SEP>' + head_entity
        input_id = tokenizer.encode(input_token, add_special_tokens=False)
        tries+=1
    input_id = input_id[:input_len]
    input_id += [tokenizer.convert_tokens_to_ids('<PAD>')] * (input_len - len(input_id))
    return torch.tensor([input_id], dtype=torch.long)

def connect_entities_wcontains(head_entity, tail_entity, withentity, temperature=1, num_outs=1, top_k=0, top_p=1.0):
    gen_input = prepare_input_wcontains(head_entity, tail_entity, withentity)
    gen_input = gen_input.to(device)
    gen_output, prob_arr = generator(gen_input, temperature=temperature, num_outs=num_outs, top_k=top_k, top_p=top_p)
    prob_sum = [-sum(math.log(x) for x in l if x < 1 and x >= 0) for l in prob_arr]
    outs = []
    for gen in gen_output:
        path = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
        path = ' '.join(path.replace('<PAD>', '').split())
        try:
            out = path[path.index('<SEP>') + 6:]
        except:
            print('weird path', path)
            out = path[16:]

        outs.append(out)
    return outs, prob_sum


def clean_keywords(target):
    keywords = kw_extractor.extract_keywords(target)
    targetyake_candidates = [kw[0].lower() for kw in keywords]
    # target_candidates = [x for x in nlp(target) if x.is_stop!=True and len(x)>2 and x.text in targetyake_candidates]
    targetyake_candidates = sorted(targetyake_candidates, key=len, reverse=True)
    target_candidates = []
    for t in targetyake_candidates:
        found = False
        for x in target_candidates:
            if t in x.split():
                found = True
                break
        if not found:
            target_candidates.append(t)

    return target_candidates


def augment_datapoint_otters(data, write_file, verbose=False, remove_overlap=False, split_entities_into_multi=False):
    global skip_overlap_cnt
    global augmentation_cnt
    with open(write_file, 'w') as outfile:
        ottersdata = []
        for datarow in tqdm(data):
            index = datarow[0]
            context = datarow[1].strip()
            context_words = context.split()
            # if len(context_words) > 250 - 100:
            #     context = ' '.join(context_words[-150:])
            target = datarow[2].strip()
            output = datarow[3].strip()
            if context == 'context' and target=='target':
                continue
            # print(responses, '**', output)
            dp = {'context': context, 'target': target, 'response': output}
            # print('\n'+ ' [context] '+context+' [response] :-- original target', target)
            # target_keywords = clean_keywords(target)
            # context_keywords = clean_keywords(context)
            # response_keywords = clean_keywords(output)

            target_keywords = get_verbnouns(target)
            context_keywords = get_verbnouns(context)
            response_keywords = get_verbnouns(output)
            # print('target_keywords', target_keywords)
            # print('context_keywords', context_keywords)
            # print('response_keywords', response_keywords)
            dp['target_keywords'] = target_keywords
            dp['context_keywords'] = context_keywords
            dp['response_keywords'] = response_keywords
            if verbose:
                print(dp)
            dp['paths'] = dict()

            if split_entities_into_multi:
                def _augment(lst):
                    global augmentation_cnt
                    for w in lst[:]:
                        tmp = w.strip().split()
                        if len(tmp)>1:
                            lst.extend(tmp)
                            augmentation_cnt += len(tmp)
                # print("earlier : context_words = ", context_words)
                _augment(context_keywords)
                # print("after aug : context_words = ", context_words)
                _augment(target_keywords)

            for head_entity in context_keywords:
                for tail_entity in target_keywords:
                    if tail_entity in ['person'] or head_entity in ['person'] or 'not' in tail_entity:
                        continue

                    # check if want to remove head/tail
                    # - remove head-tail where they overlap (eat, eat food)
                    if remove_overlap and check_overlap(head_entity, tail_entity):
                        skip_overlap_cnt+=1
                        continue

                    dp['paths'][head_entity + '---' + tail_entity] = dict()
                    dpht = dp['paths'][head_entity + '---' + tail_entity]
                    paths, scores = connect_entities(head_entity, tail_entity, temperature=0.7, num_outs=5, top_k=0,
                                                     top_p=0.9)
                    dpht['headtotail_paths'] = paths
                    dpht['headtotail_scores'] = scores
                    if verbose:
                        print('\n', head_entity, '->', tail_entity)
                        for i, path in enumerate(paths):
                            print(path, scores[i])
                    # for withentity in response_keywords:
                    ctx_keywordstouse = [x for x in context_keywords if x not in head_entity and x not in tail_entity]
                    tgt_keywordstouse = [x for x in target_keywords if x not in head_entity and x not in tail_entity]
                    response_keywords = [x for x in response_keywords if x not in head_entity and x not in tail_entity]
                    keywordstouse = list(set(ctx_keywordstouse + tgt_keywordstouse + response_keywords))
                    dpht['augkeywords_intermediate'] = dict()
                    dphtwe = dpht['augkeywords' + '_intermediate']
                    paths, scores = connect_entities_wcontains(head_entity, tail_entity, keywordstouse, temperature=0.7,
                                                               num_outs=5, top_k=0, top_p=0.9)
                    dphtwe['withentity_paths'] = paths
                    dphtwe['withentity_scores'] = scores
                    dphtwe['entities_inpath'] = keywordstouse
                    if verbose:
                        print('withentity ', keywordstouse, ' gold keywords', response_keywords)
                        for i, path in enumerate(paths):
                            print(path, scores[i])

                    dpht['goldkeywords_intermediate'] = dict()
                    dphtwe = dpht['goldkeywords' + '_intermediate']
                    paths, scores = connect_entities_wcontains(head_entity, tail_entity, response_keywords,
                                                               temperature=0.7, num_outs=5, top_k=0, top_p=0.9)
                    dphtwe['withentity_paths'] = paths
                    dphtwe['withentity_scores'] = scores
                    dphtwe['entities_inpath'] = response_keywords

                    if verbose:
                        print('withentity ', response_keywords)
                        for i, path in enumerate(paths):
                            print(path, scores[i])
                        # paths, scores = connect_entities(head_entity, withentity, temperature=1,num_outs=5,top_k=0,top_p=0.9)
                        # dphtwe['headtoentity_paths'] = paths
                        # dphtwe['headtoentity_scores'] = scores
                        # if verbose:
                        #     print('\nTwo-step paths', head_entity, withentity, tail_entity)
                        #     for i, path in enumerate(paths):
                        #         print(path, scores[i])
                        # paths, scores = connect_entities(withentity, tail_entity, temperature=1,num_outs=5,top_k=0,top_p=0.9)
                        # dphtwe['entityto_paths'] = paths
                        # dphtwe['entityto_scores'] = scores
                        # if verbose:
                        #     for i, path in enumerate(paths):
                        #         print(path, scores[i])
            json.dump(dp, outfile)
            outfile.write('\n')
            outfile.flush()

    return ottersdata

## default options
remove_overlap = False
split_entities_into_multi = False
outfile_suffix=''

remove_overlap = True
split_entities_into_multi = True

## file names
if remove_overlap:
    outfile_suffix += '.removeoverlap' # ''
if split_entities_into_multi:
    outfile_suffix += '.splitentities'  # ''



infile = '../../data/simple/in_domain_test_out.csv'
start_index = 0
end_index = 2
data= get_csv_lines(infile)
print("--- total data size = ", len(data))
data = data[start_index:end_index]
path = './ottersout/'
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
write_file = './ottersout/augfp4_'+ infile.split('/')[-1]+'_'+str(start_index)+'-'+str(end_index)+outfile_suffix+'.jsonl'
print('Start augmentation')


new_data = augment_datapoint_otters(data,write_file,
                                    verbose=False,
                                    remove_overlap=remove_overlap,
                                    split_entities_into_multi=split_entities_into_multi)

print("skip_overlap_cnt = ", skip_overlap_cnt) # skip_overlap_cnt =  18
print("augmentation_cnt = ", augmentation_cnt)
