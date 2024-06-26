# -*- coding: utf-8 -*-
"""
@author: Alperen Canbey
"""

#%% Set up selenium and chrome

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import os
import random
import bs4 as BeautifulSoup
import re
import docx
import datetime

# @juan
import pandas as pd

#%% Chrome settings
chrome_options = Options()
chrome_options.headless = False
chrome_options.add_argument('--incognito')
chrome_options.add_argument("--window-size=1920,1200")

## Set the browser
path = r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
#path = r'C:\Users\u109898\Documents\chromedriver_win32\chromedriver.exe'
browser = webdriver.Chrome(options = chrome_options, executable_path = path)

#%% Query settings
# @juan
# Suggestion: Use [] to initialize an empty array (works a bit faster)
titlelist  = [] 
textlist   = []
sourcelist = []
datelist   = []
queries    = []

# @juan
# Since the code is running in incognito google requires
# accepting terms and conditions
browser.get("http://www.google.com")
browser.find_elements_by_class_name('jyfHyd')[1].click()


#%%
#news_df = pd.read_csv('news.csv', sep='/t')
news_df = pd.read_excel('news.xlsx')
company_df = pd.read_excel('company.xlsx', sheet_name = 'Sheet1')

start_date = datetime.datetime.strptime('2010-01-01', "%Y-%m-%d").date()
end_date = datetime.date.today().strftime("%Y-%m-%d")
end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
delta = (end_date - start_date).days

#page = 0 
#d = 1235
#query = "google inc"

try:
    for index,row in company_df.iterrows():
        query = row['name']
        start_day = row['check']
        if start_day >= delta:
            pass
        print(query)
    
        for d in range(start_day,delta,7):
            
            time.sleep(1)
            titles = "start"
            page = 0
        
            print(f"Scraping news for {d}-{d+7}")
            while len(titles) > 0:
                
                link = 'https://www.google.com/search?q={}&hl=en&tbs=cdr:1,cd_min:1/{}/2010,cd_max:1/{}/2010&tbm=nws&start={}'.format(query,d,d+6,page)
                browser.get(link)
        
            
                #%% Crawler process 
                
                soup = BeautifulSoup.BeautifulSoup(browser.page_source, 'html.parser')
                content = soup.prettify()
                
                """ it is better to look at content and the structure of html file when doing web crawler
                """
                 
                ## Find titles
                #titles = re.findall('<div aria-level="2" class="JheGif nDgy9d" role="heading" style="-webkit-line-clamp:2">\n\ +(.+)', content)
                titles = soup.find_all("div", {"class": "JheGif nDgy9d"})
                for i in range(0,len(titles)):
                    titles[i] = titles[i].text
    
                
                # texts = re.findall('<div class="Y3v8qd">', content)
                texts = soup.find_all("div", {"class": "Y3v8qd"})
                for i in range(0,len(texts)):
                    texts[i] = texts[i].text
    
                source = re.findall('</g-img>\n\ +(.+)', content)
                sources = [source[i] for i in range(len(source)) if source[i].find('</div>')==-1 and source[i].find('<g-img>')==-1 and source[i].find('Languages')==-1]
                
                date = re.findall('<span>\n\ +(.+)', content)
                dates = [date[i] for i in range(len(date)) if date[i].find('Languages')==-1 and date[i].find('<em>')==-1 and date[i].find(':')==-1]
                
                qs = [query for s in sources]
                titlelist  += titles
                textlist   += texts
                sourcelist += sources
                datelist   += dates
                queries    += qs
                
                # @juan
                page = page + 10
                
                if page > 20:
                    break
                 
                time.sleep(2)
            
            company_df.at[index, 'check'] = d + 7
            company_df.to_excel('company.xlsx', index=False)
except Exception as e:
    print('Error : ', e)
    dict = {'title': titlelist, 'text': textlist, 'source': sourcelist, 'date': datelist, 'company': queries}
    df=pd.DataFrame(dict)
    news_df = news_df.append(df)
    news_df.to_excel("news.xlsx", index=False)
        
            
                

            
            
            
            
    
#%%
#dict = {'title': titlelist, 'text': textlist, 'source': sourcelist, 'date': datelist, 'company': queries}
#df=pd.DataFrame(dict)
#news_df = news_df.append(df)
#news_df.to_csv("news3.csv", index=False)

#news_df=pd.DataFrame(dict)
#dict = {'title': titlelist, 'text': textlist, 'source': sourcelist, 'date': datelist, 'company': queries}
#df=pd.DataFrame(dict)
#news_df = news_df.append(df)
#news_df.to_excel("news.xlsx", index=False)



#csv de 3 satır olduğu içn excele yazıyorum 
#titles ı da değiştirdim 
#error verince code çalışması denenmedi 
#take the unique news at the end. since there will be repitations coming from the interrupted scraping 



#company listesi kontrol et 

# To be done:
#     1. Turn to next page: maybe looking for something related to "next"...
#     2. Loop over all pages, and generate a random break among loops
#     3. Loop over all products of interest
    
#     6. translate issues
#     7. titles text source ve text aynı olmadığında warning ver
# """






# %%
