# -*- coding: utf-8 -*-
"""
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


today = datetime.date.today()
end_date = datetime.datetime(today.year, today.month, today.day)

news=[]
tickers = []

company_df = pd.read_csv(r'C:\Users\Alperen Canbey\Downloads\cik_ticker.csv')

tickers = company_df.ticker[0:len(company_df)]
company = company_df



#company.drop(index = company.index[company['ticker']=="chl"].tolist())






with open('dataset_unique_balanced.pickle', 'rb') as output_train_unique_balanced:
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



#with open('SVM_trained.pickle', 'wb') as svm:
#    pickle.dump(classifier_linear, svm)
        
with open('SVM_trained.pickle', 'rb') as svm_:
    classifier_linear = pickle.load(svm_)
    
    

classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()






company = company_df
company = company_df[5399:]



sentiment_results = []

for day in range(0,380):
    date = end_date - timedelta(430)
    d_time = date + timedelta(day)
    d = d_time.strftime("%Y-%m-%d")
    print(d)
   
    
    for index,row in company.iterrows():
        ticker = row['ticker']
        cik = row['cik']
        print(ticker)
        print(index)
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
        
        sentiment_results.append((d_time, d, ticker, cik, p, neu, neg))
        


#with open('daily_sentiment_dataset_v1.pickle', 'wb') as daily_sentiment:
#   pickle.dump(sentiment_results, daily_sentiment)

with open('daily_sentiment_dataset_v1.pickle', 'rb') as daily_sentiment_:
    sentiment_results2 = pickle.load(daily_sentiment_)
 
    
sentiment_results = sentiment_results2
    
sentiment_results_df = pd.DataFrame(list(sentiment_results))
sentiment_results_df.columns = ['date_format','date','ticker','p','neu','neg']

#company.at[company.index[company['ticker']=="tot"].tolist(),"ticker"] = "tot.to"

#sentiment_results_df.at[company.index[company['ticker']=="tot"].tolist(),"ticker"] = "tot.to"

import numpy as np


sentiment_results_df["sentiment1"] = ( sentiment_results_df['p'] / (sentiment_results_df['p'] + sentiment_results_df['neg']) ) *2 - 1
sentiment_results_df["sentiment1"][sentiment_results_df["sentiment1"].isnull()] = 0

sentiment_results_df["sentiment2"] = np.log( (1+sentiment_results_df['p']) / (1+sentiment_results_df['neg']) )
#sentiment_results_df["sentiment2"][sentiment_results_df["sentiment2"].isnull()] = 0
#sentiment_results_df["sentiment2"][np.isinf(sentiment_results_df["sentiment2"])] = 50

sentiment_results_df["total_news"] = sentiment_results_df['p'] + sentiment_results_df['neu'] + sentiment_results_df['neg']
sentiment_results_df["dispersion"] = abs(sentiment_results_df['p'] - sentiment_results_df['neg'])


#ATM implied vol etkisine d ebak


stock_info = pd.DataFrame()

import pandas_datareader as pdr

company = company_df[0:2010]
company = company[1297:]


for index,row in company.iterrows():
    ticker = row['ticker']
    print(index)
    print(ticker)
    
    if ticker in tickers_lowercase:
        stock = pdr.get_data_yahoo(row['ticker'], 
                              start=datetime.datetime(2021, 5, 1), 
                              end=datetime.datetime(2021, 7, 15))
        
        stock["Close_lagged"] = stock["Close"].shift(1)
        
        stock["return_stock"] =  (stock["Close"] - stock["Close_lagged"])*100/stock["Close_lagged"]
        
        stock["return_lag1"] = stock["return_stock"].shift(-1)
        stock["return_lag2"] = stock["return_stock"].shift(-2)
        stock["return_lag3"] = stock["return_stock"].shift(-3)
        stock["return_lag4"] = stock["return_stock"].shift(-4)
        stock["return_lag5"] = stock["return_stock"].shift(-5)
        
        stock["volume_lag1"] = stock["Volume"].shift(-1)
        stock["volume_lag2"] = stock["Volume"].shift(-2)
        stock["volume_lag3"] = stock["Volume"].shift(-3)
        stock["volume_lag4"] = stock["Volume"].shift(-4)
        stock["volume_lag5"] = stock["Volume"].shift(-5)
    
        stock["date"] = stock.index.strftime("%Y-%m-%d")
        stock["ticker"] = ticker
        
        stock_info = pd.concat([stock_info, stock])
        
    else: 
        print(f"Requested data got no results for company {ticker}")
        
        
        
    

info1 = stock_info.groupby(['ticker'] , as_index=False).agg('mean')
info_vol = pd.DataFrame()
info_vol = pd.DataFrame(info1["ticker"])
info_vol[2] = round(info1["Volume"])
info_vol.columns = ["ticker", "avg_volume"]


stock_info4 = pd.merge(stock_info, info_vol, on = ('ticker'))

#with open('stock_price_data1.pickle', 'wb') as yahoo_data1:
#    pickle.dump(stock_info4, yahoo_data1)    

    
sentiment_data = pd.merge(sentiment_results_df, stock_info4, on = ('date', 'ticker'))

sentiment_data.to_stata("sentiment_dataset_daily2000.dta")

    

stock_info2 = pd.DataFrame()

import pandas_datareader as pdr
company = company_df[0:2010]

company = company[1297:]
for index,row in company.iterrows():
    ticker = row['ticker']
    print(index)
    print(ticker)
    
    if ticker in tickers_lowercase:
        
        stock = pdr.get_data_yahoo(row['ticker'], 
                              start=datetime.datetime(2021, 5, 1), 
                              end=datetime.datetime(2021, 7, 15))
        
        stock["Close_lagged1"] = stock["Close"].shift(-1)
        stock["return_1days"] =  (stock["Close_lagged1"] - stock["Close"])*100/stock["Close"]
        
        stock["Close_lagged2"] = stock["Close"].shift(-2)
        stock["return_2days"] =  (stock["Close_lagged2"] - stock["Close"])*100/stock["Close"]
        
        stock["Close_lagged3"] = stock["Close"].shift(-3)
        stock["return_3days"] =  (stock["Close_lagged3"] - stock["Close"])*100/stock["Close"]
        
        stock["Close_lagged4"] = stock["Close"].shift(-4)
        stock["return_4days"] =  (stock["Close_lagged4"] - stock["Close"])*100/stock["Close"]
        
        stock["Close_lagged5"] = stock["Close"].shift(-5)
        stock["return_5days"] =  (stock["Close_lagged5"] - stock["Close"])*100/stock["Close"]
        
        stock["volume_1days"] = stock["Volume"].shift(-1)
        stock["volume_2days"] = stock["Volume"].shift(-2) + stock["Volume"].shift(-1)
        stock["volume_3days"] = stock["Volume"].shift(-3) + stock["Volume"].shift(-2) + stock["Volume"].shift(-1)
        stock["volume_4days"] = stock["Volume"].shift(-4) + stock["Volume"].shift(-3) + stock["Volume"].shift(-2) + stock["Volume"].shift(-1)
        stock["volume_5days"] = stock["Volume"].shift(-5) + stock["Volume"].shift(-4) + stock["Volume"].shift(-3) + stock["Volume"].shift(-2) + stock["Volume"].shift(-1)
    
        stock["date"] = stock.index.strftime("%Y-%m-%d")
        stock["ticker"] = ticker
        
        stock_info2 = pd.concat([stock_info2, stock])
    
    else: 
        print(f"Requested data got no results for company {ticker}")
        

#with open('stock_price_data.pickle', 'wb') as yahoo_data:
#    pickle.dump(stock_info2, yahoo_data)    


info1 = stock_info2.groupby(['ticker'] , as_index=False).agg('mean')
info_vol = pd.DataFrame()
info_vol = pd.DataFrame(info1["ticker"])
info_vol[2] = round(info1["Volume"])
info_vol.columns = ["ticker", "avg_volume"]
stock_info3 = pd.merge(stock_info2, info_vol, on = ('ticker'))


sentiment_data2 = pd.merge(sentiment_results_df, stock_info3, on = ('date', 'ticker'))

sentiment_data2.to_stata("sentiment_dataset_daily_2nd_2000.dta")





    

sentiment_results_df["week_number"] = sentiment_results_df["date_format"].apply(lambda x: datetime.date(x.year, x.month, x.day).isocalendar()[1])
sentiment_results_weekly = sentiment_results_df.groupby(['week_number', 'ticker'] , as_index=False).agg('sum')


sentiment_results_weekly["sentiment1"] = ( sentiment_results_weekly['p'] / (sentiment_results_weekly['p'] + sentiment_results_weekly['neg']) ) *2 - 1
sentiment_results_weekly["sentiment1"][sentiment_results_weekly["sentiment1"].isnull()] = 0

sentiment_results_weekly["sentiment2"] = np.log( (1+sentiment_results_weekly['p']) / (1+sentiment_results_weekly['neg']) )
#sentiment_results_weekly["sentiment2"][sentiment_results_weekly["sentiment2"].isnull()] = 0
#sentiment_results_weekly["sentiment2"][np.isinf(sentiment_results_weekly["sentiment2"])] = 50

sentiment_results_weekly["total_news"] = sentiment_results_weekly['p'] + sentiment_results_weekly['neu'] + sentiment_results_weekly['neg']
sentiment_results_weekly["dispersion"] = abs(sentiment_results_weekly['p'] - sentiment_results_weekly['neg'])




stock_week =  stock_info.iloc[:, 0:20]
stock_week["week_number"] = stock_week["date"].apply(lambda x: datetime.date(datetime.datetime.strptime(x, '%Y-%m-%d').date().year, datetime.datetime.strptime(x, '%Y-%m-%d').date().month ,datetime.datetime.strptime(x, '%Y-%m-%d').date().day).isocalendar()[1])


date_list = list(stock_week.groupby('week_number', as_index=False).agg('max')["date"])
calculate_return = stock_week[ stock_week["date"].apply(lambda x: x in date_list) ]


calculate_return.drop(index = calculate_return.index[calculate_return['ticker']=="rol"].tolist())

calculate_return = calculate_return.iloc[:,:][calculate_return['ticker']!="rol"]
calculate_return = calculate_return.iloc[:,:][calculate_return['ticker']!="cvac"]
calculate_return = calculate_return.iloc[:,:][calculate_return['ticker']!="mpwr"]


stock_info_weekly = pd.DataFrame()
count = 0
for  stock in list(set(calculate_return["ticker"])):
    print(stock)
    temp = calculate_return[calculate_return["ticker"] == stock]
    temp["Close_lagged"] = temp["Close"].shift(1)
    temp["return_stock_weekly"] =  (temp["Close"] - temp["Close_lagged"])*100/temp["Close_lagged"]
    
    temp["return_stock_weekly_lag1"] = temp["return_stock_weekly"].shift(-1)
    temp2 = temp.iloc[:, 19:23]
    
    temp_vol = stock_week[stock_week["ticker"] == stock]
    weekly_volume = temp_vol.groupby(['week_number', 'ticker'], as_index=False).agg('sum')["Volume"]
    temp2["weekly_volume"] = list(weekly_volume)
    temp2["weekly_volume_lag1"] = temp2["weekly_volume"].shift(-1)
    
    if count == 0:
        stock_info_weekly = pd.concat([stock_info_weekly, temp2])
        count = count +1 
        
    elif len(stock_info_weekly.iloc[1,:]) == len(temp2.iloc[1,:]):
        stock_info_weekly = pd.concat([stock_info_weekly, temp2])

    else:
        print("nanannsnsnsnsn")


stock_info_weekly2 = pd.merge(stock_info_weekly, info_vol, on = ('ticker'))


#with open('stock_price_data_weekly.pickle', 'wb') as yahoo_data_weekly:
#    pickle.dump(stock_info_weekly2, yahoo_data_weekly)    
    



sentiment_data_weekly = pd.merge(sentiment_results_weekly, stock_info_weekly2, on = ('week_number', 'ticker'))
sentiment_data_weekly.to_stata("sentiment_dataset_weekly_2000.dta")
