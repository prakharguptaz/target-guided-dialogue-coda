import pandas as pd    
from rake_nltk import Rake
import json
import csv
import logging
import psutil
cpu_count = psutil.cpu_count()
import sklearn
import nltk
import spacy
# import claucy                                                                                                                                               
import neuralcoref
from allennlp.predictors.predictor import Predictor
# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
nlp = spacy.load('en_core_web_lg')  
neuralcoref.add_to_pipe(nlp)
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
from isanlp.annotation import Event, TaggedSpan, Sentence, Token
from isanlp.annotation_repr import CSentence
import re

class ProcessorSrlAllennlp:
    def __init__(self, model_path=None, model = None):
        if model:
            self._predictor = model
            return
        if model_path:
            self._predictor = Predictor.from_path(model_path)
        else:
            # self._predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
            self._predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
          # self._predictor = Predictor.from_path("https://allennlp.s3.amazonaws.com/models/srl-model-2020.02.10.tar.gz")
          # https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz

    def _find_object_start(self, seq, start=0):
        def is_start(idx):
            parts = seq[idx].split('-', maxsplit=1)
            if len(parts) < 2:
                return False

            label, _ = parts
            return label == 'B'

        i = start
        while i < len(seq) and not is_start(i):
            i += 1

        if i >= len(seq):
            return -1, ''

        return i, seq[i].split('-')[1]

    def _find_object_end(self, seq, obj_start_idx):
        i = obj_start_idx + 1
        while i < len(seq) and not seq[i].startswith('B-') and seq[i] != 'O':
            i += 1

        return i

    def _convert_format(self, allennlp_srl):
        events = []
        for verb in allennlp_srl['verbs']:
            event = Event(pred=None, args=[])
            
            i = 0
            while True:
                start, tp = self._find_object_start(verb['tags'], start=i)
                if start == -1:
                    break

                finish = self._find_object_end(verb['tags'], start)
                if tp == 'V':
                    event.pred = (start, finish)
                else:
                    arg = TaggedSpan(tag=tp, begin=start, end=finish)
                    event.args.append(arg)

                i = finish

            events.append(event)

        return events       
    
    def print_args_event(self, event, tokens):
        verb = ' '.join(tokens[e].text for e in range(event.pred[0], event.pred[1]))
        print('Verb: ', verb)
        
        arg_res = []
        for arg in event.args:
            arg_text = ' '.join(tokens[e].text for e in range(arg.begin, arg.end))
            print('Arg: ', arg.tag, arg_text)
            if arg.tag[-1] == "0": 
                arg_res = ["<A0> " + arg_text] + arg_res
            elif arg.tag[-1] == "1" and len(arg_res) == "0":
                arg_res = ["<A1> " + arg_text] + arg_res
            else: # append arg1 or arg2 to end
                if arg.tag[-1] == "1" or arg.tag[-1] == "2":
                    arg_res += ["<A" + arg.tag[-1] +"> " + arg_text]
        
        final_arg_string = "<V> " + verb
        for arg in arg_res[:2]:
            final_arg_string += " " + arg
        print(final_arg_string)

    
#     def get_args_event(self, event, tokens, print_srl = False):
#         verb = ' '.join(tokens[e].text for e in range(event.pred[0], event.pred[1]))
#         if print_srl:
#             print(verb)
#         arg_res = []
#         for arg in event.args:
#             arg_text = ' '.join(tokens[e].text for e in range(arg.begin, arg.end))
#             if print_srl:
#                 print('Arg: ', arg.tag, arg_text)
#             if arg.tag[-1] == "0": 
#                 arg_res = ["<A0> " + arg_text] + arg_res
#             elif arg.tag[-1] == "1" and len(arg_res) == 0:
#                 arg_res = ["<A1> " + arg_text] + arg_res
#             else: # append arg1 or arg2 to end
#                 if arg.tag[-1] == "1" or arg.tag[-1] == "2":
#                     arg_res += ["<A" + arg.tag[-1] +"> " + arg_text]
        
#         final_arg_string = "<V> " + verb
#         for arg in arg_res[:2]:
#             final_arg_string += " " + arg
        
#         return (final_arg_string)
    
#     def get_args_event(self, event, tokens, print_srl = False):
#         verb = ' '.join(tokens[e].text for e in range(event.pred[0], event.pred[1]))
#         if print_srl:
#             print(verb)
#         arg_res = []
#         final_arg_string = ''
#         arg_dict = dict()
#         for arg in event.args:
#             arg_text = ' '.join(tokens[e].text for e in range(arg.begin, arg.end))
#             if print_srl:
#                 print('Arg: ', arg.tag, arg_text)
#             prevtagvalue = arg_dict.get(arg.tag, '')
#             if prevtagvalue=='':
#                 arg_dict[arg.tag] = arg_text
#             else:
#                 arg_dict[arg.tag] = prevtagvalue + ' ' +  arg_text
        
#         arg0 = arg_dict.pop('ARG0', '')
#         argm = arg_dict.pop('ARGM', '')
#         if argm!='': argm = ' ' + argm
#         if arg0!='':
#             final_arg_string = arg0 + argm + ' ' + verb
#         else:
#             arg1 = arg_dict.pop('ARG1', '')
#             final_arg_string = arg1 + argm + ' ' + verb       
#         for arg, val in arg_dict.items():
#             final_arg_string += " " + val
            
#         return (final_arg_string)
    
    def get_args_event(self, event, tokens, print_srl = False):
        if event.pred is None:
            return ''
        verb = ' '.join(tokens[e].text for e in range(event.pred[0], event.pred[1]))
        if print_srl:
            print(verb)
        arg_res = []
        final_arg_string = ''

        tags_present = set()
        for arg in event.args:
            arg_text = ' '.join(tokens[e].text for e in range(arg.begin, arg.end))
            tags_present.add(arg.tag)
        if len(tags_present)<2:
            return ''
        verb_encountered = False
#         for arg, val in arg_dict.items():
        for arg in event.args:
            val = ' '.join(tokens[e].text for e in range(arg.begin, arg.end))
            if final_arg_string == '':
                final_arg_string+=val
            else:
                final_arg_string += " " + val
            if (arg.tag in ['ARG1','ARG0', 'ARG2', 'ARGM']) and verb_encountered == False:
                final_arg_string = final_arg_string + ' ' + verb
                verb_encountered = True
            
#         print('final_arg_string:', final_arg_string)
        return (final_arg_string)

    def get_actobj_event(self, event, tokens, print_srl = False):
        if event.pred is None:
            return ''
        verb = ' '.join(tokens[e].text for e in range(event.pred[0], event.pred[1]))
        if print_srl:
            print(verb)
        arg_res = []
        final_arg_string = ''

        if verb in ['am', "'s", "is", 'have', 'has', 'had', 'be', 'are', 'do', "'m"]:
            return ''

        for arg in event.args:
            val = ' '.join(tokens[e].text for e in range(arg.begin, arg.end))
            if (arg.tag in ['ARG1']):
                ddoc = nlp(verb + ' ' + val)
                obj = []
                lastfound = -1
                for i,x in enumerate(ddoc):
                    if x.pos_ == "NOUN" or x.pos_ == "PROPN" or x.pos_=="PRON" and x.text not in ['it', 'they', 'i']:
                        if lastfound==i-1 or lastfound==-1:
                            obj.append(x.text)
                            lastfound = i 
                    if x.pos_ == 'VERB':
                        verb = x.lemma_
                        # break
                    # else:
                    #     if lastfound!=-1:
                    #         break
                if len(obj)==0:
                    return ''
                final_arg_string = verb + ' ' + ' '.join(obj)
                break
            
#         print('final_arg_string:', final_arg_string)
        return (final_arg_string)
    
    def get_sentence_args(self, srl_sent, sentence):
        args_list = []
        for event in srl_sent:
            sentence_tokenized = [Token(x.text) for x in sentence]
            arg_string = self.get_args_event(event, CSentence(sentence_tokenized, Sentence(0, len(sentence_tokenized))))
#             if '<A' in arg_string:
#                 args_list.append(arg_string)
            args_list.append(arg_string)
            
        return args_list

    def get_sentence_actobj(self, srl_sent, sentence):
        args_list = []
        for event in srl_sent:
            sentence_tokenized = [Token(x.text) for x in sentence]
            arg_string = self.get_actobj_event(event, CSentence(sentence_tokenized, Sentence(0, len(sentence_tokenized))))
#             if '<A' in arg_string:
#                 args_list.append(arg_string)
            if arg_string!='':
                args_list.append(arg_string)
            
        return args_list
        
    def get_args(self, results, sentences_coref):
        for sent_num, srl_sent in enumerate(results):
            for event in srl_sent:
                sentence_tokenized = [Token(x) for x in sentences_coref[sent_num]]
                self.get_args_event(event, CSentence(sentence_tokenized, Sentence(0, len(sentence_tokenized))))
    
    def get_tokenized_sentence(self, sentence):
        sentence_tokenized = [Token(x) for x in sentence.split(' ')]
        
        return sentence_tokenized
    
    def get_tokenized_sentences(self, sentence):
        all_tokenized_sentences = []
        sentence_tokenized_doc = nlp(sentence)
        for tokenized_sentence in sentence_tokenized_doc.sents:
            all_tokenized_sentences.append(tokenized_sentence)

        return all_tokenized_sentences

    def get_pure_srl(self, sentences, print_srl = False):
        all_results = []
        for sent in sentences:
            result = []
            sentence_text = sent
            split_tokenized_sentences = self.get_tokenized_sentences(sentence_text)
            for sentence in split_tokenized_sentences:
                sentence_tokenized = [e.text for e in sentence]
                allennlp_srl = self._predictor.predict_tokenized(
                                    tokenized_sentence = sentence_tokenized
                                )
                if print_srl:
                    print(sentence_tokenized)
                    print(allennlp_srl)
    #             allennlp_srl = self._predictor.predict(sentence = sentence)
                events = self._convert_format(allennlp_srl)
                args_list = self.get_sentence_args(events, sentence)
                result+=args_list
            all_results.append(result)   
                
        return all_results
    
    def get_formatted_srl(self, sentences, print_srl = False):
        all_results = []
        for sent in sentences:
            result = []
            sentence_text = sent[0]
            sentence_ent_map = sent[1]
            split_tokenized_sentences = self.get_tokenized_sentences(sentence_text)
            for sentence in split_tokenized_sentences:
                sentence_tokenized = [e.text for e in sentence]
                allennlp_srl = self._predictor.predict_tokenized(
                                    tokenized_sentence = sentence_tokenized
                                )
                if print_srl:
                    print(sentence_tokenized)
                    print(allennlp_srl)
    #             allennlp_srl = self._predictor.predict(sentence = sentence)
                events = self._convert_format(allennlp_srl)
                args_list = self.get_sentence_args(events, sentence)
                result+=args_list
            all_results.append(result)   
                
        return all_results

    def get_formatted_srlactobj(self, sentences, print_srl = False):
        all_results = []
        all_actobj= []
        for sent in sentences:
            result = []
            actobjres = []
            sentence_text = sent[0]
            sentence_ent_map = sent[1]
            split_tokenized_sentences = self.get_tokenized_sentences(sentence_text)
            for sentence in split_tokenized_sentences:
                sentence_tokenized = [e.text for e in sentence]
                allennlp_srl = self._predictor.predict_tokenized(
                                    tokenized_sentence = sentence_tokenized
                                )
                if print_srl:
                    print(sentence_tokenized)
                    print(allennlp_srl)
    #             allennlp_srl = self._predictor.predict(sentence = sentence)
                # import pdb;pdb.set_trace()
                events = self._convert_format(allennlp_srl)
                args_list = self.get_sentence_args(events, sentence)
                ao_list = self.get_sentence_actobj(events, sentence)
                actobjres+=ao_list
                result+=args_list
            all_results.append(result)   
            all_actobj.append(actobjres)
                
        return all_results, all_actobj

def resolve_coref(sentences):
    result = []
    for sentence in sentences:
        sentence_doc = nlp(sentence)
        sentence_coref_resolved = (sentence_doc._.coref_resolved)
        result.append(sentence_coref_resolved)
        
    return result


def replace_NER(doc):
    map_ent = {}
    sentence = doc.text
    for ent in doc.ents:
        i = len(map_ent)
        map_ent['ent' + str(i)] = ent.text
        try:
            sentence = re.sub(ent.text, 'ent' + str(i), sentence)
        except:
            pass
        
    return sentence, map_ent

def replace_post_NER(doc, map_ent):
    #first replace pronouns
    pl = [['i', 'my', 'me', 'mine', 'we', 'us', 'our', 'ours'], ['you', 'your', 'yours'], [ "he", "she", "it", "him", "her", "it", "his", "her", "its", "his","hers","its", 'they', 'them', 'their', 'theirs']]
    base_pronouns = ['I', 'you', 'they']
    pl, base_pronouns = [], []
    pronoun_dict = {}
    for i, subpl in enumerate(pl):
        for l in subpl:
              pronoun_dict[l.lower()] = base_pronouns[i]
    #print(pronoun_dict)  
    pronoun_indices = {}
    sentence = ""
    for token in doc:
        token_text = token.text
        if token_text.lower() in pronoun_dict:
            token_text = pronoun_dict[token_text.lower()]
            if token_text in pronoun_indices:
                token_text = pronoun_indices[token_text]
            else:
                i = len(map_ent)
                map_ent['ent' + str(i)] = token_text
                pronoun_indices[token_text] = 'ent' + str(i)
                token_text = 'ent' + str(i)

        sentence += token_text + ' '
    sentence = sentence.strip()
    
#     sentence = doc.text
    for ent in doc.ents:
        i = len(map_ent)
        map_ent['ent' + str(i)] = ent.text
        try:
            sentence = re.sub(ent.text, 'ent' + str(i), sentence)
        except:
            pass
        
    return sentence, map_ent

def replace_coref_cluster(doc):
    sentence_coref_resolved = (doc._.coref_resolved)
    map_ent = {}
    for i, cluster in enumerate(doc._.coref_clusters):
        main = cluster.main.text
        map_ent["ent" + str(i)] = cluster.main.text
        sentence_coref_resolved = re.sub(main, "ent" + str(i), sentence_coref_resolved)
#     print(sentence_coref_resolved)
#     print(map_ent)
    
    return sentence_coref_resolved, map_ent

# def replace_with_corefs(sentences):
#     results = []
#     for sentence in sentences:
#         doc = nlp(sentence)
#         res = replace_coref_cluster(doc)
#         doc = nlp(res[0])
#         res = replace_post_NER(doc, res[1])
#         results.append(res)
    
#     return results

def replace_with_corefs(sentences):
    results = []
    for sentence in sentences:
        sentence = sentence.replace('_eos', '')
        doc = nlp(sentence)
#         res = replace_coref_cluster(doc)
        coref_resolved = doc._.coref_resolved
#         print(doc)
#         print(coref_resolved)
#         doc = nlp(coref_resolved)
#         res = replace_post_NER(doc, {})
#         results.append(res)
        results.append((coref_resolved, {}))
    
    return results

def replace_with_corefs(sentences):
    results = []
    for sentence in sentences:
        sentence = sentence.replace('_eos', '')
        doc = nlp(sentence)
#         res = replace_coref_cluster(doc)
        coref_resolved = doc._.coref_resolved
#         print(doc)
#         print(coref_resolved)
#         doc = nlp(coref_resolved)
#         res = replace_post_NER(doc, {})
#         results.append(res)
        results.append((coref_resolved, {}))
    
    return results

def process_da(data_point, type_agenda):
    #sample data_point OrderedDict([('', '0'), ('Unnamed: 0', '0'), ('data_id', '0'), ('context', 'hey man , you wanna buy some weed ?'), 
    #('response_num', '0'), ('response', 'some what ?'), ('context_last', 'hey man , you wanna buy some weed ?'), 
    #('prediction', 'Signal-non-understanding'), ('act_obj', 'what some what'), ('rake_phrases', '')])

    data_point['agenda'] = []
    da = data_point['prediction']
    # if type_agenda == 'act_obj':
    #     actobj = data_point['act_obj'].split('; ') if \
    #                         "; " in data_point['act_obj'] \
    #                          else \
    #                         [data_point['act_obj']]
    #     if len(actobj) ==1 and actobj[0] == '': actobj = []
    # if type_agenda = 'rake_phrases':
    #     actobj = data_point['rake_phrases'].split('; ') if \
    #                         "; " in data_point['rake_phrases'] else \
    #                         [data_point['rake_phrases']]
    #     if len(actobj) ==1 and actobj[0] == '': actobj = []

    if type(data_point[type_agenda]) == list:
        actobj = data_point[type_agenda]
    else:
        actobj = data_point[type_agenda].split('; ') if "; " in data_point[type_agenda] \
                             else \
                            [data_point[type_agenda]]
    if len(actobj) ==1 and actobj[0] == '': actobj = []
    actobj = [x for x in actobj if x!='']
    response = data_point['response']
    data_point['agenda_' + type_agenda] = []
    # import pdb;pdb.set_trace()
    #https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
    map_act = {'Action-directive': 'Suggest', 'Quotation': 'Quote', 'Statement-opinion': 'Give opinion', 'Statement-non-opinion':'State fact',\
                'Yes Answers': 'Agree and answer', 'Tag-Question': 'Question', 'Affirmative Non-yes Answers': 'Answer positively', 'Agree/Accept':'Agree', 'Conventional-opening' : 'Greet' ,\
                'Rhetorical-Question': 'Ask rhetorically', 'Thanking': 'Thank', 'Open-Question': 'Ask open-ended question', 'Conventional-closing': 'Greet',\
                 'Offers, Options Commits': 'Commit or Offer', 'Apology': 'Apologize', 'Negative Non-no Answers': 'Answer negatively', 'Other Answers': 'Answer',\
                 'Maybe/Accept-part':'Accept partially','Summarize/Reformulate': 'Summarize'}

    if da == 'Signal-non-understanding' and len(response)>1 and response[-1]=='?':
        for ao in actobj:
            data_point['agenda_' + type_agenda].append('Ask clarify : ' + ao)
    
    elif da == 'Yes-No-Question' or da == 'Declarative Yes-No-Question':
        # print(actobj)
        # print(keyphrases)
        # print('--')
        for ao in actobj:
            data_point['agenda_' + type_agenda].append('Ask if : ' + ao)

    elif da == 'Wh-Question' or da == 'Declarative Wh-Question':
        wh_type = response.split(' ')[0]
        for ao in actobj:
            # data_point['agenda_' + type_agenda].append('ask ' + wh_type + ' ' + ao)
            data_point['agenda_' + type_agenda].append('Ask : ' + ao)

    elif da in map_act:
        for ao in actobj:
            data_point['agenda_' + type_agenda].append(map_act[da] + ' : ' + ao)
    
    else:
        for ao in actobj:
            data_point['agenda_' + type_agenda].append(da + ' : ' + ao)

    # data_point['agenda'] = '; '.join(data_point['agenda'])
    # if len(data_point['agenda'])==0: data_point['agenda'] = 'no agenda'


def process_file(filename, outfilename):

    data_points = []
    with open(filename, 'r') as data_file:
        for line in csv.DictReader(data_file): 
            # print(line)
            data_points.append(line) 

    processed_outputs = []
    for dnum, data_point in enumerate(data_points):
        # dnum==(list(data_point.values())[0])

        process_da(data_point)
        # print(data_point['response'], ' --- ', data_point['agenda'])

        processed_outputs.append(data_point)
        if dnum>1000: break




    outfile = open( outfilename,'w')
    w = csv.DictWriter(outfile, processed_outputs[0].keys())
    w.writeheader()
    for outdict in processed_outputs: 
        w.writerow(outdict)

    outfile.close()



if __name__ == '__main__':
    # eval_data = process_file('data/test_da_key.csv', 'data/agenda_raw/test_agenda.csv')
    print('----------')
    sentences = ["I opened my eyes. Looking to my razor-sharp claws, I found they were now neatly clipped. My ears flopped on my head lazily, too soft and formless to hunt properly. Most of all, the hunger was gone. Confusion clouded my mind and I tilted my head instinctively. I approached a nearby puddle and looked in"]
    # sentences = ["Writers often rely on plans or sketches to write long stories, but most current language models generate word by word from left to right."]
    sentences = ["are they available for travel ? Where can I learn to make dessert in Paris? Which kind it matches your taste most? let's go. do you think you can make yourself easily understood in english ?"]
    # sentences = ["what are your personal weaknesses ? _eos i 'm afraid i 'm a poor talker . i 'm not comfortable talking with the people whom i have just met for the first time . that is not very good for business , so i have been studying public speaking . _eos are you more of a leader or a follower ? _eos i do n't try to lead people . i 'd rather cooperate with everybody , and get the job done by working together ."]
    sentences = ["why do you need spotify when you can listen to free and ad - free music on the shuffle ?", " i 'm canadian but i get a few american tv channels and honestly i still think it 's weird that they have commercials for lawyers and law firms .",
                 " I am so grateful for my family due to an incident with my friend","I just rode the biggest and scariest roller coaster at Six Flags today.", "My ears flopped on my head lazily, too soft and formless to hunt properly."
                 "i really want to see this maury episode .", "i want an episode dedicated to the mooch "] 

    srl_object = ProcessorSrlAllennlp()
    #replace with corefs
    sentences_coref = replace_with_corefs(sentences)
    print('sentences_coref:', sentences_coref)
    # results = srl_object.get_formatted_srl(sentences_coref, print_srl=True)
    results, aoresults = srl_object.get_formatted_srlactobj(sentences_coref, print_srl=False)
    # sentences_coref = (sentences)
    # results = srl_object.get_pure_srl(sentences_coref)


    for i,result in enumerate(results):
        print(result)
        print('act_obj:',aoresults[i])

