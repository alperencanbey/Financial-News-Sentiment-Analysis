
import finnhub
import time
import datetime
from datetime import timedelta
finnhub_client = finnhub.Client(api_key="c4eloviad3ifs61iimpg")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle 
import numpy as np




with open('svm_preprocess_balanced.pickle', 'rb') as output_train_unique_balanced:
    training_data = pickle.load(output_train_unique_balanced)


#TRAINING THE SVM #########################

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(doc[0] for doc in training_data)

train_labels = []
for (sentence, label) in training_data:
    train_labels.append(label)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='rbf') 
classifier_linear.fit(train_vectors, train_labels)
##############################################


today = datetime.date.today()
end_date = datetime.datetime(today.year, today.month, today.day)

company_df = pd.read_csv(r'C:\Users\Alperen Canbey\Downloads\cik_ticker.csv')

company = company_df
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
        
        sentiment_results.append((d_time, d, ticker, cik, p, neu, neg))


#SAVE THE INITIAL VERSION OF THE DATASET ############################
#with open('daily_sentiment_dataset_preprocessed_v1.pickle', 'wb') as daily_sentiment2_preprocessed:
#   pickle.dump(sentiment_results, daily_sentiment2_preprocessed)

with open('daily_sentiment_dataset_preprocessed_v1.pickle', 'rb') as daily_sentiment2_preprocessed_:
   stockkk= pickle.load(daily_sentiment2_preprocessed_)  
#####################################################################




# DEFINE THE VARIABLES ##############################################################
sentiment_results_df = pd.DataFrame(list(sentiment_results))
sentiment_results_df.columns = ['date_format','date','ticker', 'cik', 'p','neu','neg']

sentiment_results_df["sentiment_daily"] = ( sentiment_results_df['p'] / (sentiment_results_df['p'] + sentiment_results_df['neg']) ) *2 - 1
sentiment_results_df["sentiment_daily"][sentiment_results_df["sentiment_daily"].isnull()] = 0

sentiment_results_df["log_sentiment_daily"] = np.log( (1+sentiment_results_df['p']) / (1+sentiment_results_df['neg']) )

sentiment_results_df["total_news"] = sentiment_results_df['p'] + sentiment_results_df['neu'] + sentiment_results_df['neg']




stock_sentiment_info = pd.DataFrame()


for index,row in company.iterrows():
    ticker = row['ticker']
    print(index)
    print(ticker)
    
    stock = sentiment_results_df[sentiment_results_df['ticker'] == ticker]
    #sorted(stock)
    
    moving_average = pd.DataFrame(np.convolve(stock["sentiment_daily"], np.ones(3)/3, mode='valid'))
    df = pd.DataFrame(np.nan, index=[0, 1], columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["sentiment_last3days"] = pd.DataFrame(df2.iloc[:,1])
    
    moving_average = pd.DataFrame(np.convolve(stock["sentiment_daily"], np.ones(5)/5, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,4), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["sentiment_last5days"] = pd.DataFrame(df2.iloc[:,1])
    
    
    moving_average = pd.DataFrame(np.convolve(stock["sentiment_daily"], np.ones(7)/7, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,6), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["sentiment_last7days"] = pd.DataFrame(df2.iloc[:,1])
    
    moving_average = pd.DataFrame(np.convolve(stock["sentiment_daily"], np.ones(10)/10, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,9), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["sentiment_last10days"] = pd.DataFrame(df2.iloc[:,1])
    
    moving_average = pd.DataFrame(np.convolve(stock["sentiment_daily"], np.ones(14)/14, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,13), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["sentiment_last14days"] = pd.DataFrame(df2.iloc[:,1])
    
    
    
    moving_average = pd.DataFrame(np.convolve(stock["log_sentiment_daily"], np.ones(3)/3, mode='valid'))
    df = pd.DataFrame(np.nan, index=[0, 1], columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["log_sentiment_last3days"] = pd.DataFrame(df2.iloc[:,1])
    
    moving_average = pd.DataFrame(np.convolve(stock["log_sentiment_daily"], np.ones(5)/5, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,4), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["log_sentiment_last5days"] = pd.DataFrame(df2.iloc[:,1])
    
    
    moving_average = pd.DataFrame(np.convolve(stock["log_sentiment_daily"], np.ones(7)/7, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,6), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["log_sentiment_last7days"] = pd.DataFrame(df2.iloc[:,1])
    
    moving_average = pd.DataFrame(np.convolve(stock["log_sentiment_daily"], np.ones(10)/10, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,9), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["log_sentiment_last10days"] = pd.DataFrame(df2.iloc[:,1])
    
    moving_average = pd.DataFrame(np.convolve(stock["log_sentiment_daily"], np.ones(14)/14, mode='valid'))
    df = pd.DataFrame(np.nan, index=range(0,13), columns=['0'])
    df2 = df.append(moving_average)
    df2.index = stock.index
    stock["log_sentiment_last14days"] = pd.DataFrame(df2.iloc[:,1])
    
    
    stock_sentiment_info = pd.concat([stock_sentiment_info, stock])
    
    
    
    
#SAVE THE FINAL VERSION OF THE DATASET ############################ 
with open('daily_sentiment_dataset_preprocessed_v2.pickle', 'wb') as daily_sentiment2_preprocessed:
   pickle.dump(stock_sentiment_info, daily_sentiment2_preprocessed)

with open('daily_sentiment_dataset_preprocessed_v2.pickle', 'rb') as daily_sentiment2_preprocessed_:
   stock_sentiment_info2 = pickle.load(daily_sentiment2_preprocessed_)  


stock_sentiment_info.to_stata("daily_sentiment_dataset_preprocessed_v2.dta")
################################################################### 