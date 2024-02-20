# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:55:34 2021

@author: Alperen Canbey
"""



import pickle
import pandas as pd
import time
import datetime
from datetime import timedelta
import numpy as np

sentiment_analysis = []
        

with open('non-unique_all.pickle', 'rb') as deneme:
    all_news_train_set = pickle.load(deneme)
    
    
for listeler in all_news_train_set:
    for news in listeler:
        sentiment_analysis.append((news["tickers"], news["date"], news["sentiment"]))
        
temp_df = pd.DataFrame(list(sentiment_analysis))
temp =  pd.DataFrame(list(temp_df.iloc[:,0]))
temp = temp.iloc[:,0:5]
temp[5] = temp_df[1]
temp[6] = temp_df[2]
temp = temp.drop_duplicates(subset = [0,1,2,3,4,5,6])

temp2 = []

for i in range(0,len(temp)):
    for j in range(0,4):
        x = temp.iloc[i,:]
        x = list(x)
        temp2.append((x[j],x[5],x[6]))
        
a = pd.DataFrame(list(temp2))
b = a[a[0].apply(lambda x: x is not None)]

b[3] = b[1].apply(lambda x: x[5:16])
b[4] = b[3].apply(lambda x: datetime.datetime.strptime(x, '%d %b %Y').date())


sentiment_analysis_df = b
sentiment_analysis_df.columns = ["ticker", "date_string", "sentiment", "date_format", "date"]

sentiment_analysis_df = sentiment_analysis_df[sentiment_analysis_df["sentiment"] != 'Neutral']

sentiment_analysis_df["total_news"] = 1
sentiment_analysis_df["total_positive"] = (sentiment_analysis_df["sentiment"] == 'Positive')*1
sentiment_analysis_df["total_negative"] = (sentiment_analysis_df["sentiment"] == 'Negative')*1
                                                                              
sentiment_analysis_df2 = sentiment_analysis_df.groupby(['date', 'ticker'], as_index=False).agg('sum')

list1 = ['FB', 'AMZN', 'TSLA', 'NFLX', 'GOOG', 'GOOGL', 'TWTR']
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'FB' ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'AMZN' ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'TSLA' ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'NFLX' ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'GOOG' ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'GOOGL' ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["ticker"] != 'TWTR' ]


sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["total_news"] > 1 ]
sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["total_positive"] + sentiment_analysis_df2["total_negative"] != 0]



sentiment_analysis_df2 = sentiment_analysis_df2[sentiment_analysis_df2["date"] >  datetime.date.today() - timedelta(320) ]
sentiment_analysis_df2["week_number"] = sentiment_analysis_df2["date"].apply(lambda x: datetime.date(x.year, x.month, x.day).isocalendar()[1])

sentiment_analysis_df2["sentiment1"] = ( sentiment_analysis_df2["total_positive"] / (sentiment_analysis_df2["total_positive"] + sentiment_analysis_df2["total_negative"]))*2 -1
sentiment_analysis_df2["sentiment1"][sentiment_analysis_df2["sentiment1"].isnull()] = 0

sentiment_analysis_df2["sentiment2"] = np.log( (1+sentiment_analysis_df2["total_positive"]) / (1+sentiment_analysis_df2["total_negative"]) )



sentiment_analysis_weekly = sentiment_analysis_df2.groupby(['week_number', 'ticker'], as_index=False).agg('sum')

sentiment_analysis_weekly["sentiment1"] = ( sentiment_analysis_weekly["total_positive"] / (sentiment_analysis_weekly["total_positive"] + sentiment_analysis_weekly["total_negative"]))*2 -1
sentiment_analysis_weekly["sentiment1"][sentiment_analysis_weekly["sentiment1"].isnull()] = 0

sentiment_analysis_weekly["sentiment2"] = np.log( (1+sentiment_analysis_weekly["total_positive"]) / (1+sentiment_analysis_weekly["total_negative"]) )


ticker_list = list(set(sentiment_analysis_df2["ticker"]))




stock_info = pd.DataFrame()

import pandas_datareader as pdr

company = pd.DataFrame(list(ticker_list))
company = company[338:]
for index,row in company.iterrows():
    ticker = row[0]
    print(index)
    print(ticker)
    stock = pdr.get_data_yahoo(ticker, 
                          start=datetime.datetime(2021, 1, 4), 
                          end=datetime.datetime(2021, 10, 30))
    
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
    
    stock_info = pd.concat([stock_info, stock])
    

stock_info["date"] = stock_info["date"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())      



info1 = stock_info.groupby(['ticker'] , as_index=False).agg('mean')
info_vol = pd.DataFrame()
info_vol = pd.DataFrame(info1["ticker"])
info_vol[2] = round(info1["Volume"])
info_vol.columns = ["ticker", "avg_volume"]
stock_info3 = pd.merge(stock_info, info_vol, on = ('ticker'))


sentiment_data_newsapi = pd.merge(sentiment_analysis_df2, stock_info3, on = ('date', 'ticker'))

sentiment_data_newsapi["date"] = sentiment_data_newsapi["date"].apply(lambda x: datetime.date(x.year, x.month, x.day).timetuple().tm_yday)

#date i stinge de dönüştür
sentiment_data_newsapi.to_stata("sentiment_dataset_daily_newsapi_new.dta")

    






stock_week =  stock_info3.iloc[:, 0:24]

stock_week["week_number"] = stock_week["date"].apply(lambda x: datetime.date(x.year, x.month ,x.day).isocalendar()[1])


date_list = list(stock_week.groupby('week_number', as_index=False).agg('max')["date"])
calculate_return = stock_week[ stock_week["date"].apply(lambda x: x in date_list) ]

stock_info_weekly = pd.DataFrame()
for  stock in list(set(calculate_return["ticker"])):
    print(stock)
    temp = calculate_return[calculate_return["ticker"] == stock]
    temp["Close_lagged"] = temp["Close"].shift(1)
    temp["return_stock_weekly"] =  (temp["Close"] - temp["Close_lagged"])*100/temp["Close_lagged"]
    
    temp["return_stock_weekly_lag1"] = temp["return_stock_weekly"].shift(-1)
    temp2 = temp.iloc[:, 21:28]
    
    temp_vol = stock_week[stock_week["ticker"] == stock]
    weekly_volume = temp_vol.groupby(['week_number', 'ticker'], as_index=False).agg('sum')["Volume"]
    temp2["weekly_volume"] = list(weekly_volume)
    temp2["weekly_volume_lag1"] = temp2["weekly_volume"].shift(-1)
    
    stock_info_weekly = pd.concat([stock_info_weekly, temp2])
    
    
    

sentiment_data_weekly_newsapi = pd.merge(sentiment_analysis_weekly, stock_info_weekly, on = ('week_number', 'ticker'))

sentiment_data_weekly_newsapi
sentiment_data_weekly_newsapi["date"] = sentiment_data_weekly_newsapi["date"].apply(lambda x: x.strftime("%Y-%m-%d"))

sentiment_data_weekly_newsapi.to_stata("sentiment_dataset_weekly_newsapi_new.dta")

    



