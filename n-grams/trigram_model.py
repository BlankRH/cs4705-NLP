import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Summer 2012 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus): 
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    l = []
    s = sequence

    s.insert(0, 'START')
    for i in range(1, n-1):
        s.insert(0, 'START')
    s.append('STOP')
    for i in range(0, len(s)-n+1):
        t = ()
        for j in range(0, n):
            t += (s[i+j],)
        l.append(t)

    return l


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)

        self.wordcnt = 0
        self.count_ngrams(generator)
        


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        for sentence in corpus:
            for word in sentence:
                self.wordcnt += 1

            l = get_ngrams(sentence, 1)
            for ngram in l:
                if ngram not in self.unigramcounts:
                    self.unigramcounts[ngram] = 1
                else:
                    self.unigramcounts[ngram] += 1

            l = get_ngrams(sentence, 2)
            for ngram in l:
                if ngram not in self.bigramcounts:
                    self.bigramcounts[ngram] = 1
                else:
                    self.bigramcounts[ngram] += 1

            l = get_ngrams(sentence, 3)
            for ngram in l:
                if ngram not in self.trigramcounts:
                    self.trigramcounts[ngram] = 1
                else:
                    self.trigramcounts[ngram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        if trigram not in self.trigramcounts:
            return 0

        return self.trigramcounts[trigram]/self.bigramcounts[(trigram[0], trigram[1])]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        if bigram not in self.bigramcounts:
            return 0

        return self.bigramcounts[bigram]/self.unigramcounts[(bigram[0],)]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  

        if unigram not in self.unigramcounts:
            return 0

        return self.unigramcounts[unigram]/self.wordcnt

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram = lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability((trigram[1], trigram[2])) + lambda3 * self.raw_unigram_probability((trigram[2],))

        return trigram
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        l = get_ngrams(sentence, 3)
        res = 0

        try: 
            for trigram in l:
                res += math.log2(self.smoothed_trigram_probability(trigram))
        except:
            print(trigram)           

        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        prob_sum = 0
        word_cnt = 0

        for sentence in corpus:
            for word in sentence: 
                word_cnt += 1
            prob_sum += self.sentence_logprob(sentence)

        l = prob_sum / word_cnt

        return 2 ** (-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1= model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if(pp1 < pp2):
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp1= model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if(pp1 > pp2):
                correct += 1
            total += 1

            # .. 
        
        return correct / total

if __name__ == "__main__":
    
    #model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment("ets_toefl_data/train_high.txt", "ets_toefl_data/train_low.txt", "ets_toefl_data/test_high", "ets_toefl_data/test_low")
    print(acc)

