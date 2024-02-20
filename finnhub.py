# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:26:00 2021

@author: Alperen Canbey
"""


import finnhub
import time
import datetime
from datetime import timedelta
finnhub_client = finnhub.Client(api_key="c4eloviad3ifs61iimpg")
import json
import pandas as pd

#%%


# %%

today = datetime.date.today()
end_date = datetime.datetime(today.year, today.month, today.day)

news=[]
tickers = []

company_df = pd.read_csv(r'C:\Users\Alperen Canbey\Downloads\cik_ticker.csv')

tickers = company_df.ticker[0:len(company_df)]


#9983 eqfn
company = company_df[0:300]


import requests

try:
    
    for index,row in company.iterrows():
        ticker = row['ticker']
        print(ticker)
    
        for s in range(355):
            d = end_date - timedelta(s)
            d = d.strftime("%Y-%m-%d")
            time.sleep(0.85)  
            c_news = finnhub_client.company_news(ticker, _from=d, to=d)
            nresults = len(c_news)
            print(f"Requested data from date {d} got {nresults} results for company {ticker}")
            with open(r'C:\Users\Alperen Canbey\dataset\{}{}.json'.format(ticker,d), 'w') as f:
                json.dump(c_news, f)
            df_json = pd.read_json(r'C:\Users\Alperen Canbey\dataset\{}{}.json'.format(ticker,d))
            #df_json.to_excel(r'C:\Users\Alperen Canbey\dataset_excel\{}{}.xlsx'.format(ticker,d))
            #df_json.to_csv(r'C:\Users\Alperen Canbey\dataset_csv\{}{}.xlsx'.format(ticker,d))
    
except requests.exceptions.Timeout:
    print("Timeout occurred")
    
    
    
    ticker = "aapl"
    
   x = datetime.datetime.a.date
a =    1637179560


for s in range(6):
    d = end_date - timedelta(s)
    d = d.strftime("%Y-%m-%d")
    time.sleep(0.85) 
    print(ticker)
    c_news = finnhub_client.company_news(ticker, _from=d, to=d)
    nresults = len(c_news)
    print(f"Requested data from date {d} got {nresults} results for company {ticker}")
    with open(r'C:\Users\Alperen Canbey\Desktop\Project A\{}{}.json'.format(ticker,d), 'w') as f:
        json.dump(c_news, f)
    df_json = pd.read_json(r'C:\Users\Alperen Canbey\Desktop\Project A\{}{}.json'.format(ticker,d))
    
    
    
    
    
    
    
    
    
import requests

try:
    
    d = end_date  - timedelta(2)
    d = d.strftime("%Y-%m-%d") 

    for index,row in company.iterrows():
        ticker = row['ticker']
        print(ticker)
        
        time.sleep(0.85)  
        c_news = finnhub_client.company_news(ticker, _from=d, to=d)
        nresults = len(c_news)
        print(f"Requested data from date {d} got {nresults} results for company {ticker}")
        with open(r'C:\Users\Alperen Canbey\\Desktop\Project A\{}{}.json'.format(ticker,d), 'w') as f:
            json.dump(c_news, f)
        df_json = pd.read_json(r'C:\Users\Alperen Canbey\Desktop\Project A\{}{}.json'.format(ticker,d))
            
except requests.exceptions.Timeout:
    print("Timeout occurred")









for s in range(3):
    d = end_date - timedelta(6)
    d = d.strftime("%Y-%m-%d")
    
    for index,row in company.iterrows():
        time.sleep(0.85) 
        ticker = row['ticker']
        print(ticker)
        c_news = finnhub_client.company_news(ticker, _from=d, to=d)
        nresults = len(c_news)
        print(f"Requested data from date {d} got {nresults} results for company {ticker}")
        with open(r'C:\Users\Alperen Canbey\dataset\{}{}.json'.format(ticker,d), 'w') as f:
            json.dump(c_news, f)
        df_json = pd.read_json(r'C:\Users\Alperen Canbey\dataset\{}{}.json'.format(ticker,d))
        #df_json.to_excel(r'C:\Users\Alperen Canbey\dataset_excel\{}{}.xlsx'.format(ticker,d))
        #df_json.to_csv(r'C:\Users\Alperen Canbey\dataset_csv\{}{}.xlsx'.format(ticker,d))
# %%
