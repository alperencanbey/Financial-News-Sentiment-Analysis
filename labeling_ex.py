# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:51:00 2021

@author: Alperen Canbey
"""


import finnhub
import time
import datetime
from datetime import timedelta
finnhub_client = finnhub.Client(api_key="c4eloviad3ifs61iimpg")
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import pickle 
from nltk.tokenize.treebank import TreebankWordDetokenizer


today = datetime.date.today()
end_date = datetime.datetime(today.year, today.month, today.day)

news=[]
tickers = []

company_df = pd.read_csv(r'C:\Users\Alperen Canbey\Downloads\cik_ticker.csv')

tickers = company_df.ticker[0:len(company_df)]

ticker = 'tsla'



    
    
sentiment_results2 = []

for day in range(0,61):
    date = end_date - timedelta(199)
    d_time = date + timedelta(day)
    d = d_time.strftime("%Y-%m-%d")
    print(d)
   
    
    df_json = pd.read_json(r'C:\Users\Alperen Canbey\dataset\{}{}.json'.format(ticker,d))
    news_text = []
    
    
    for i in range(0, len(df_json)):
        #print(df_json["summary"][i])
        if isinstance(df_json["summary"][i], str):
            sentence = (df_json["headline"][i] + ": " + df_json["summary"][i])
            news_text.append(sentence)
        else:
            sentence = df_json["headline"][i]
            news_text.append(sentence)
    
    
    if len(news_text) > 0 :
        
        news_text_processed = []

        for sentence in news_text:
            sentence = text_preprocessing(sentence)
            clean_sentence = mark_negation(sentence)
            final_sentence = TreebankWordDetokenizer().detokenize(clean_sentence)
            news_text_processed.append(final_sentence)
    
        test_vectors = vectorizer.transform(doc for doc in news_text_processed)
        
        prediction_linear = classifier_linear.predict(test_vectors)
        t2 = time.time()        
        
        p = len(prediction_linear[prediction_linear == "Positive"])
        neu = len(prediction_linear[prediction_linear == "Neutral"])
        neg = len(prediction_linear[prediction_linear == "Negative"])
    else:
        p= 0
        neu=0
        neg = 0
    
    nresults = len(news_text)
    
    print(f"Requested data from date {d} got {nresults} results for company {ticker}.")
    print(f"For company {ticker} the daily sentiment is {p} positive news, {neu} neutral news and {neg} negative news.")
    
    sentiment_results2.append((d_time, d, ticker, p, neu, neg))
