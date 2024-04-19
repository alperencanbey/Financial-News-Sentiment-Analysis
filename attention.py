# -*- coding: utf-8 -*-
"""
@author: Alperen Canbey
"""

#https://towardsdatascience.com/a-very-precise-fast-way-to-pull-google-trends-data-automatically-4c3c431960aa
# ya ticker aranınca direk finansallar gelsin. scrapingle yap ya da sal 

import pandas as pd
import pytrends
from pytrends.request import TrendReq
import time
pytrend = TrendReq()
#import enchant
from nltk.corpus import words
import nltk
nltk.download('words')
import statistics
import pickle


russel1 = pd.read_csv(r'C:\Users\Alperen Canbey\Desktop\Sentiment RA\russell1000_20210628.csv')
russel2 = pd.read_csv(r'C:\Users\Alperen Canbey\Desktop\Sentiment RA\russell1000_20200629.csv')
russel_list = pd.concat([russel1, russel2])
russel_list_unique = russel_list.drop_duplicates(subset=['ticker'])


company_df = pd.read_csv(r'C:\Users\Alperen Canbey\Downloads\cik_ticker.csv')

russel = list()



russel_list_unique = company_df


russel_list_unique = russel_list_unique[4083:]


#tickerlar büyük yazıldıysa .lower() ekle

for index,row in russel_list_unique.iterrows():
    print(row.ticker)
    print(row.ticker.lower() in words.words())
    if (row.ticker.lower() in words.words()) == False:
        print("x")
        russel.append(row.ticker.lower())
        

russel_u = list(set(russel))


russel2 = pd.DataFrame(russel)
russel3 = pd.DataFrame(russel).drop_duplicates()


with open('russell1000_tickers.pickle', 'wb') as trends4:
    pickle.dump(russel, trends4) 
    
    
with open('trendsdata_tickers.pickle', 'wb') as trends4:
    pickle.dump(russel3, trends4) 

with open('trendsdata_tickers.pickle', 'rb') as trends5:
     russel = pickle.load(trends5)
    
with open('russell1000_tickers.pickle', 'rb') as trends_russel:
    russel3 = pickle.load(trends_russel)


russel1 = russel.to_list()


keywords = russel_u
keywords = russel_u[7616:]


EXACT_KEYWORDS = keywords
DATE_INTERVAL='2010-01-01 2022-01-01'
#COUNTRY=["WW"] #Use this link for iso country code
COUNTRY=["US"]
#select category "News" --> "Financial Markets"
CATEGORY=1163 # Use this link to select categories
SEARCH_TYPE='' #default is 'web searches',others include 'images','news','youtube','froogle' (google shopping)



Individual_EXACT_KEYWORD = list(zip(*[iter(EXACT_KEYWORDS)]*1))
Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]
#dicti = {}

i = 7617
    
for Country in COUNTRY:
    for keyword in Individual_EXACT_KEYWORD:
        print(i-1)
        print(keyword) 
        pytrend.build_payload(kw_list=keyword,
                              timeframe = DATE_INTERVAL, 
                              geo = Country, 
                              cat=CATEGORY,
                              gprop=SEARCH_TYPE) 
        dicti[i] = pytrend.interest_over_time()
        i+=1
        time.sleep(2) 
        df_trends = pd.concat(dicti, axis=1)



    
df_trends.columns = df_trends.columns.droplevel(0) #drop outside header
df_trends = df_trends.drop('isPartial', axis = 1) #drop "isPartial"
df_trends.reset_index(level=0,inplace=True) #reset_index
#df_trends.columns=['date','AAPL-US','TSLA-US','AMZN-US','PLTR-US','MRVL-US','AAPL-UK','TSLA-UK','AMZN-UK','PLTR-UK','MRVL-UK','AAPL-Germany','TSLA-Germany','AMZN-Germany','PLTR-Germany','MRVL-Germany'] #change column names
df_trends.rename(columns={'index': 'date'}, inplace=True)


#with open('google_trends_russell1000_v2.pickle', 'wb') as trends:
#    pickle.dump(df_trends, trends)    


#with open('google_trends_russell1000_v2.pickle', 'rb') as trend2:
#    df_trends2 = pickle.load(trend2)

#from 2010
#with open('trends_df_trends.pickle', 'wb') as trends3:
#    pickle.dump(df_trends, trends3)    

import math

log_trends = df_trends
for i in range(0,len(df_trends)):
    for j in range(1,len(df_trends.columns)):
        if df_trends.iloc[i,j] != 0:
            log_trends.iloc[i,j] = math.log(df_trends.iloc[i,j])
        else:
            log_trends.iloc[i,j] = 0

#with open('log_google_trends_russell1000_v2.pickle', 'wb') as trends2:
#    pickle.dump(df_trends, trends2)   


#with open('log_google_trends_russell1000.pickle', 'rb') as trend2:
#    log_trends = pickle.load(trend2)

#from2010
#with open('log_google_trends_russell1000_from2010.pickle', 'wb') as trends4:
#    pickle.dump(df_trends, trends4)   



import numpy as np

myvec = np.arange(1, 4517*145+1)
asvi = pd.DataFrame(myvec.reshape(-1, 4517))
for i in range(8,len(df_trends)):
    print(i)
    for j in range(1,len(df_trends.columns)):
        asvi.iloc[i,j] = log_trends.iloc[i,j] - statistics.median(log_trends.iloc[(i-8):i,j])


asvi = asvi.drop(asvi.index[0:8])
asvi.columns = log_trends.columns
asvi.date = log_trends.date


asvi["date"] = asvi["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
asvi.index = asvi["date"]
asvi = asvi.drop(columns=['date'])
asvi2 = asvi.T
asvi2["ticker"] = asvi2.index 
asvi3 = pd.merge(asvi2, company_df, on = ('ticker'))

asvi4 = pd.melt(asvi3,id_vars=['ticker', "cik"],var_name='date', value_name='attention')


with open('ASVI_russell1000_v2.pickle', 'wb') as trends3:
    pickle.dump(asvi, trends3) 
    
    
    
with open('ASVI_russell1000_v2.pickle', 'rb') as trends_russel_:
    asvi = pickle.load(trends_russel_)
    



with open('ASVI_long_from2010.pickle', 'wb') as trends4:
    pickle.dump(asvi4, trends4) 
    
    
    
with open('ASVI_russell1000_v2.pickle', 'rb') as trends_russel_:
    asvi = pickle.load(trends_russel_)


asvi4.to_stata("ASVI_long_from2010.dta")

    



import seaborn as sns
sns.set(color_codes=True)
dx = df_trends.plot(figsize = (12,8),x="date", y=['AAPL-US','AMZN-US','TSLA-US'], kind="line", title = "Google Trends")
dx.set_xlabel('Date')
dx.set_ylabel('Trends Index')
dx.tick_params(axis='both', which='both', labelsize=10)

