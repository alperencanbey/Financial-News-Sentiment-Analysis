# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:52:09 2021

@author: Alperen Canbey
"""



import finnhub
import time
import datetime
from datetime import timedelta
finnhub_client = finnhub.Client(api_key="insert your API key for Finnhub")
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import pickle 


today = datetime.date.today()
end_date = datetime.datetime(today.year, today.month, today.day)

news=[]
tickers = []

company_df = pd.read_csv(r'C:\Users\Alperen Canbey\Downloads\cik_ticker.csv')

tickers = company_df.ticker[0:len(company_df)]
#company = company_df[0:300]

with open('dataset_unique.pickle', 'rb') as output_train_unique_balanced:
    training_data = pickle.load(output_train_unique_balanced)

train_labels = []
#training_set = []

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)


train_vectors = vectorizer.fit_transform(doc[0] for doc in training_data)

for (sentence, label) in training_data:
    train_labels.append(label)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='rbf')
t0 = time.time()


with open('SVM_trained.pickle', 'rb') as svm_:
    classifier_linear = pickle.load(svm_)
    

classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()

sentiment_results = []

#ticker = "tdc"

for day in range(15):
    d = end_date - timedelta(day)
    d = d.strftime("%Y-%m-%d")

    
    for index,row in company.iterrows():
        ticker = row['ticker']
        print(ticker)
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
            test_vectors = vectorizer.transform(doc for doc in news_text)
            
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
        
        sentiment_results.append((d, ticker, p, neu, neg))


#test_labels = []
#testing_set = []

for (sentence, label) in classification.testing_set:
    test_labels.append(label)

time_linear_train = t1-t0
time_linear_predict = t2-t1


# results

print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(test_labels, prediction_linear, output_dict=True)
print('accuracy: ', report['accuracy'])
print('positive: ', report['Positive'])
print('negative: ', report['Negative'])
print('neutral: ', report['Neutral'])



sentiment_results = []
#ticker = "tdc"
for day in range(15):
    d = end_date - timedelta(day)
    d = d.strftime("%Y-%m-%d")

    print(ticker)
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
        test_vectors = vectorizer.transform(doc for doc in news_text)
        
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
    
    sentiment_results.append((d, ticker, p, neu, neg))
