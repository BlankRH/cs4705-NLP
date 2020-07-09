#!/usr/bin/env python
import sys

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import tensorflow

import gensim
import transformers 

import string

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    synonyms = []
    for lexeme in wn.lemmas(lemma, pos):
        syn = lexeme.synset()
        for l in syn.lemmas():
            word = l.name()
            word = word.replace('_', ' ')
            if word != lemma and word not in synonyms:
                synonyms.append(word)
    return synonyms

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos

    synonyms = dict()
    for lexeme in wn.lemmas(lemma, pos):
        syn = lexeme.synset()
        for l in syn.lemmas():
            word = l.name().lower()
            word = word.replace('_', ' ')
            if word == lemma:
                continue
            if word not in synonyms:
                synonyms[word] = l.count()
            else:
                synonyms[word] += l.count()

    if len(synonyms) == 0:
        return 'smurf'
    return max(synonyms, key=lambda k: synonyms[k])

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = stopwords.words('english')
    lemma = context.lemma
    pos = context.pos
    d = dict()
    c = set()
    wnl = WordNetLemmatizer()

    for word in context.left_context:
        c.add(wnl.lemmatize(word.lower()))

    for word in context.right_context:
        c.add(wnl.lemmatize(word.lower()))

    tmp = c.copy()

    for w in c:
        if w in stop_words or not w.isalnum():
            tmp.discard(w)

    c = tmp.copy()

    for lexeme in wn.lemmas(lemma, pos=pos):
        syn = lexeme.synset()
        definition = syn.definition()
        def_words = tokenize(definition)
        if syn not in d:
            d[syn] = set()

        for word in def_words:
            d[syn].add(wnl.lemmatize(word))

        for example in syn.examples():
            ex_words = tokenize(example)
            for word in ex_words:
                d[syn].add(wnl.lemmatize(word))

        for hyp in syn.hypernyms():
            hyp_definition = hyp.definition()
            hyp_def_words = tokenize(hyp_definition)

            for word in hyp_def_words:
                d[syn].add(wnl.lemmatize(word))

            for example in hyp.examples():
                ex_words = tokenize(example)
                for word in ex_words:
                    d[syn].add(wnl.lemmatize(word))

        t = d[syn].copy()

        for w in d[syn]:
            if w in stop_words or not w.isalnum():
                t.discard(w)

        d[syn] = t.copy()

    score = dict()
    for lexeme in wn.lemmas(lemma, pos=pos):
        syn = lexeme.synset()
        for l in syn.lemmas():
            if l.name() == lemma:
                continue
            if (syn, l) not in score:
                score[(syn, l)] = 0
            score[(syn, l)] = max(score[(syn, l)], 1000 * len(d[syn] & c) + lexeme.count()*100 + l.count())

    if len(score) == 0:
        return 'smurf'
    return max(score, key=lambda k: score[k])[1].name()

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        lemma = context.lemma
        pos = context.pos
        synonyms = get_candidates(lemma, pos)
        sim = dict()
        for synonym in synonyms:
            synonym = synonym.replace(' ', '_')
            try:
                sim[synonym] = self.model.similarity(synonym, lemma)
            except:
                continue

        if len(sim) == 0:
            return 'smurf'
        return max(sim, key=lambda k: sim[k]) # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        wnl = WordNetLemmatizer()
        synonyms = get_candidates(context.lemma, context.pos)
        best_words =self.get_best_words(context)
        for word in best_words:
            word = wnl.lemmatize(word)
            word = word.replace('_', ' ')
            if word in synonyms:
                return word
        return 'smurf' # replace for part 5

    def get_best_words(self, context : Context) -> list:

        input_toks = []
        for w in context.left_context:
            input_toks.append(w.lower())
        m_index = len(input_toks)
        input_toks.append('[MASK]')
        
        for w in context.right_context:
            input_toks.append(w.lower())

        input_ids = self.tokenizer.convert_tokens_to_ids(input_toks)
        input_mat = np.array(input_ids).reshape((1,-1))
        
        outputs = self.model.predict(input_mat)
        prediction = outputs[0]
        best_words = np.argsort(prediction[0][m_index])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)

        return best_words

    def part6(self, context : Context) -> str:
        wnl = WordNetLemmatizer()
        synonyms = get_candidates(context.lemma, context.pos)
        best_words = self.get_best_words(context)

        p = wn_simple_lesk_predictor(context)

        for word in best_words:
            word = wnl.lemmatize(word)
            word = word.replace('_', ' ')
            if word not in synonyms:
                continue
            if p in best_words and best_words.index(p) - best_words.index(word) <= 50:
                return p
            else:
                return word
            
        return 'smurf' # replace for part 5
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        
        
        #prediction = wn_frequency_predictor(context) #part2
        #prediction = wn_simple_lesk_predictor(context) #part 3
        #prediction = predictor.predict_nearest(context) #part 4
        prediction = predictor.predict(context) #part 5
        
        #prediction = predictor.part6(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    
    #print(get_candidates('slow','a'))

