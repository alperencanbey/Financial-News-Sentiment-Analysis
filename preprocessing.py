import csv
import json
from nltk.tokenize import word_tokenize
from classification import Classification

import pandas as pd
#import nltk
#nltk.download('punkt')

# TODO: NaiveBayes nasil calisir?
# TODO: Ornek implementasyona neler eklenebilir, additional feature extraction for SentimentAnalyzer?
# Dataset birleştiriceksen labelları ayarla
#uniqueler kalsın üst üste binmiş olabilie
# \u00ae bunlara çözüm bul 
#ticker isimlerini yasakla

#kullanılan tickerları bul isimlerini yasakla

#IB isimlerini de
# -------------------------------
# Read Kaggle data

training_sentences = []
with open("C:/Users/Alperen Canbey/Downloads/train_data.csv", "r", encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for row in reader:
        training_sentences.append(row)

training_sentences2 = pd.read_excel("C:/Users/Alperen Canbey/Desktop/Sentiment RA/stock_data_clean.xlsx")
training_sentences2  = training_sentences2.values.tolist()
#header işini hallet

training_sentences = training_sentences + training_sentences2


for i, (label,sentence) in enumerate(training_sentences):
    training_sentences[i] = (sentence, label.title())

training_sentences= training_sentences + output_train

labels2 = []
training_set2 = []

for i in range (0,len(training_sentences2)):
    if any(s in training_sentences2[i][1] for s in ('1;')):
        labels2.append('positive')
        training_set2.append(training_sentences2[i][0])
    else:
        labels2.append('negative')
        training_set2.append(training_sentences[i][0])




labels = []
training_set = []
for i in range (0,len(training_sentences2)):
    if (training_sentences2[i][1] == 'Neutral'):
        labels.append('neutral')
    elif (training_sentences2[i][1] == 'Positive'):
        labels.append('positive')
    elif training_sentences2[i][1] == 'Negative':
        labels.append('negative')

        
training_sentences = tokenized_bozuk  


labels = []
training_set = []
k= 0
p = 0
n=0


for i in range (0,len(training_data)):
    if (training_data[i][1] == 'Neutral') and (k <7026):
        labels.append('neutral')
        k = k+1
        training_set.append(training_data[i])
    elif (training_data[i][1] == 'Positive') and (n <7026):
        labels.append('positive')
        n=n+1
        training_set.append(training_data[i])
    elif (training_data[i][1] == 'Negative') and (p <7026):
        labels.append('negative')
        training_set.append(training_data[i])
        p = p+1

        
labels.count('neutral')
labels.count('positive')
labels.count('negative')

training_sentences = training_set


with open('dataset_unique_balanced_v2.pickle', 'wb') as da:
   pickle.dump(training_set, da)

with open('dataset_unique_balanced_v2.pickle', 'rb') as da_:
   training_set2 = pickle.load(da_)  
   
   
   

from classification import Classification

import pickle

with open('dataset_unique_balanced.pickle', 'rb') as output_train_unique_balanced:
    training_data2 = pickle.load(output_train_unique_balanced)
    
    
training_sentences = training_data
training_sentences = deneme3
training_sentences = deneme3[20000:]

training_sentences = all_stocknewsapi_tokenized

import string
import re

def clean_text(text):
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace

a = clean_text(training_sentences)


from nltk.tokenize import RegexpTokenizer
import string
import re

tokenizer = RegexpTokenizer(r'\w+')
#tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')


#for i, (label, sentence) in enumerate(training_sentences):
for i, (sentence, label) in enumerate(training_sentences):
    #tokenized_sentence = word_tokenize(sentence.lower())
    sentence = re.sub(r'\d+', '', sentence)
    tokenized_sentence = tokenizer.tokenize(sentence.lower())
    training_sentences[i] = (tokenized_sentence, label)



#with open('tokenized_all.pickle', 'wb') as output_token:
#   pickle.dump(all_stocknewsapi_tokenized, output_token)


#with open('dateset_all_combined.pickle', 'wb') as output_sentences:
#   pickle.dump(training_sentences, output_sentences)
# -------------------------------
# Train classifier with training data from Kaggle

classification = Classification()
classification.set_datasets(training_sentences, split=0.8)
classification.get_dataset_features()
classification.train_classifier()
print(classification.evaluate())

# -------------------------------
# Read news data and classify


test_news =[]
with open("C:/Users/Alperen Canbey/Desktop/Sentiment RA/aapl2021-09-20.json", "r") as f:
    test_news = json.load(f)




test_news =[]
test_news = pd.read_excel("C:/Users/Alperen Canbey/Desktop/Sentiment RA/sentiment_test.xlsx")
test_news = test_news.values.tolist()
    

testing_set = []
test_labels =[]
for i, (headline, summary, label) in enumerate(test_news):
    print(headline)
    if isinstance(summary, str):
        
        sentence = headline +': ' + summary
        testing_set.append(sentence)
        test_labels.append(label.lower())
    else:
        #sentence = re.sub(r'\d+', '', headline)
        sentence = headline
        testing_set.append(sentence)
        test_labels.append(label.lower())






for example in training_sentences2:
    print(example)
    classification.testing_set = example
     #classification.testing_features = classification.sentiment_analyzer.apply_features(classification.testing_set)
    prediction = classification.sentiment_analyzer.classify(classification.testing_set[0])
    print(prediction)



classification.testing_set = news_test_set
classification.testing_features = classification.sentiment_analyzer.apply_features(classification.testing_set)
print(classification.sentiment_analyzer.evaluate(classification.testing_features))   





news_test_set = []

for news in test_news:
    sentence = news["headline"]
    #+ " " + news["summary"]
    tokenized_sentence = word_tokenize(sentence.lower())
    news_test_set.append((tokenized_sentence, "negative"))
    
    
for news in test_news:
    #tokenized_sentence = word_tokenize(sentence.lower())
    sentence = re.sub(r'\d+', '', news["headline"]  + news["summary"])
    tokenized_sentence = tokenizer.tokenize(sentence.lower())
    news_test_set.append((tokenized_sentence, "negative"))

news_test_set = training_sentences[200:220]


for example in news_test_set:
    print(example)
    classification.testing_set = example
     #classification.testing_features = classification.sentiment_analyzer.apply_features(classification.testing_set)
    prediction = classification.sentiment_analyzer.classify(classification.testing_set[0])
    print(prediction)
    
    classification.testing_set = training_sentences2
    classification.testing_features = classification.sentiment_analyzer.apply_features(classification.testing_set)
    print(classification.sentiment_analyzer.evaluate(classification.testing_features))
    classification.sentiment_analyzer.classify()
    
from nltk.sentiment import SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer()
features = sentiment_analyzer.apply_features(training_sentencess)
sentiment_analyzer.evaluate(features)