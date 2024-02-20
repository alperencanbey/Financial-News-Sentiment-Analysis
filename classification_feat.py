# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:16:24 2021

@author: Alperen Canbey
"""

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split

import nltk
from nltk.collocations import *

import pickle

class Classification:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.training_set = None
        self.training_features = None
        self.testing_set = None
        self.testing_features = None

    def set_datasets(self, training_data, split=0.8):
        training_set, testing_set = train_test_split(training_data, train_size=split)
        self.training_set = training_set
        self.testing_set = testing_set

    def get_dataset_features(self, min_frequency=3):
        #all_words = self.sentiment_analyzer.all_words([doc for doc in self.training_set])
        
        ignored_last = [".",  ":", "?", "%", "&", "'", "you.s", "N", "e", "g","a","t", "i","v", "P","o","s","u","r","l"]
        all_words_negative = self.sentiment_analyzer.all_words([mark_negation(doc[0]) for doc in self.training_set])
        all_words_negative = [w for w in all_words_negative if w not in ignored_last]
        
        unigram_features = self.sentiment_analyzer.unigram_word_feats(all_words_negative, min_freq=min_frequency)
        bigram_features = self.sentiment_analyzer.bigram_collocation_feats([mark_negation(doc[0]) for doc in self.training_set], min_freq=min_frequency)
        
        #bigram_measures = nltk.collocations.BigramAssocMeasures()
        #finder = BigramCollocationFinder.from_words(all_words_negative)

        #finder.apply_word_filter(lambda w: len(w) < 1)
        
        #finder.apply_freq_filter(3)
        #finder.nbest(bigram_measures.pmi, 10)
        #scored = finder.score_ngrams(bigram_measures.raw_freq)
        #bigram_features2 = sorted(bigram for bigram, score in scored)
        
        
        
        #trigram_measures = nltk.collocations.TrigramAssocMeasures()
        #finder_tri = TrigramCollocationFinder.from_words(all_words_negative)
        #finder_tri.apply_freq_filter(1)
        #scored = finder_tri.score_ngrams(trigram_measures.raw_freq)
        #trigram_features = sorted(trigram for trigram, score in scored)
        
        self.sentiment_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
        self.sentiment_analyzer.add_feat_extractor(extract_bigram_feats, bigrams= bigram_features)
        
        self.training_features = self.sentiment_analyzer.apply_features(self.training_set)
        self.testing_features = self.sentiment_analyzer.apply_features(self.testing_set)

    def train_classifier(self):
        self.sentiment_analyzer.train(NaiveBayesClassifier.train, self.training_features)

    def evaluate(self):
        return self.sentiment_analyzer.evaluate(self.testing_features)


#classifier.show_most_informative_features(5)
