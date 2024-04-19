# -*- coding: utf-8 -*-
"""
@author: Alperen Canbey
"""

import requests
import json
import pickle
import random
import time

#news_train_set = []
all_news_train_set = []
news_train_set_data = []
        


for page_num in range(98,198):
    response_API = requests.get('https://stocknewsapi.com/api/v1/category?section=alltickers&items=50&page={}&token=fm12u45narqytxkirgtqkel6ipthjolmze9kgrur'.format(page_num))
    #print(response_API.status_code)
    data = response_API.text
    parse_json = json.loads(data)
    
    train_set = parse_json['data']
    
    all_news_train_set.append(train_set)
    
    time.sleep(random.randint(1,20))
        
    for news in train_set:
        if isinstance(news["text"], str):
            sentence = news["title"] + "; " + news["text"]
            #tokenized_sentence = word_tokenize(sentence.lower())
            news_train_set_data.append((sentence, news["sentiment"]))
        else:
            sentence = news["title"]
            #tokenized_sentence = word_tokenize(sentence.lower())
            news_train_set_data.append((sentence, news["sentiment"]))
        
        
   
        
#my_dataset.pickle


#with open('non-unique_all.pickle', 'wb') as output_train_listoflist:
#    pickle.dump(all_news_train_set, output_train_listoflist)
        
with open('non-unique_all.pickle', 'rb') as deneme:
    all_news_train_set = pickle.load(deneme)
    
for listeler in all_news_train_set:
    for news in listeler:
        if isinstance(news["text"], str):
            sentence = news["title"] + ": " + news["text"]
            news_train_set_data.append((sentence, news["sentiment"]))
        else:
            sentence = news["title"]
            news_train_set_data.append((sentence, news["sentiment"]))


#with open('bozulan.pickle', 'wb') as output_train_tokenized:
#    pickle.dump(news_train_set, output_train_tokenized)
    
with open('bozulan.pickle', 'rb') as over_tokenized:
    bozuk_data = pickle.load(over_tokenized)
    
tokenized_bozuk = bozuk_data[:13983]
tokenized_bozuk1 = set(tokenized_bozuk)

newList=[]
for i, (liste, sentiment) in tokenized_bozuk:
    for j, (liste2, sentiment2) in newList:
    if liste[3] not in newList:
        newList.append(i)

with open('dataset_unique.pickle', 'rb') as output_train:
    output_train = pickle.load(output_train)
    
#with open('dataset_non-unique.pickle', 'wb') as output_train:
#   pickle.dump(news_train_set_data, output_train)
        
#unique_data = list(set(news_train_set_data))


#with open('dataset_unique.pickle', 'wb') as output_train_unique:
#   pickle.dump(unique_data, output_train_unique)


#with open('dataset_unique_balanced.pickle', 'wb') as output_train_unique_balanced:
#   pickle.dump(training_set, output_train_unique_balanced)

with open('dataset_unique.pickle', 'rb') as output_train_unique_balanced:
    training_data = pickle.load(output_train_unique_balanced)
    
    
    
training_data = list(set(news_train_set_data))
 
training_data = output_train

        
labels = []
for i in range (0,len(training_data)):
    if (training_data[i][1] == 'Neutral'):
        labels.append('neutral')
    elif (training_data[i][1] == 'Positive'):
        labels.append('positive')
    elif training_data[i][1] == 'Negative':
        labels.append('negative')


labels.count('neutral')
labels.count('positive')
labels.count('negative')





