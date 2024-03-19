# NLP Project Report on Financial News Sentiment Analysis

## Overview

This project aims to analyze the impact of financial news sentiment on stock market movements. By leveraging a combination of natural language processing (NLP) and machine learning techniques, we've developed a pipeline that processes financial news data to predict stock market trends.

## 1. Data Collection

Data is primarily sourced from StockNewsAPI and Finnhub, focusing on financial news articles and stock price information. These APIs provide a rich dataset that includes article content and publication dates.

## 2. Preprocessing

Preprocessing steps include:
- **Tokenization:** Splitting text into individual words or tokens.
- **Normalization:** Standardizing text by converting to lowercase.
- **Stop Words Removal:** Eliminating common words that add little value to analysis.
- **Stemming/Lemmatization:** Reducing words to their root form.
- **Negation Handling:** Words following "not" are labeled as negated to capture the change in sentiment meaning, an important step for accurately understanding the sentiment in financial news where the context can significantly alter the sentiment conveyed.

## 3. Feature Extraction

Key techniques:

### TF-IDF (Term Frequency-Inverse Document Frequency)

Assesses word importance by considering their frequency across documents. It reduces the influence of frequently occurring words and highlights rare words.

TF-IDF consists of two components:

- **Term Frequency (TF):** This measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:

<img width="355" alt="Screenshot 2024-03-19 at 18 09 14" src="https://github.com/alperencanbey/Financial-News-Sentiment-Analysis/assets/88103433/58ffbc0b-f1e2-4b32-b001-9382f6d10f4e">

- **Inverse Document Frequency (IDF):** This measures the importance of the term across a set of documents. The idea behind IDF is that terms that appear frequently in one document but not in many documents across the set are likely more important. Thus, the IDF of a rare term is high, whereas the IDF of a frequent term is likely to be low. Mathematically, IDF is calculated as:

<img width="346" alt="Screenshot 2024-03-19 at 18 09 26" src="https://github.com/alperencanbey/Financial-News-Sentiment-Analysis/assets/88103433/80d44af4-b31a-4451-817d-a492d7bea735">


Combining these two measures, TF-IDF is calculated as:

<img width="216" alt="Screenshot 2024-03-19 at 18 09 34" src="https://github.com/alperencanbey/Financial-News-Sentiment-Analysis/assets/88103433/f215026a-6e44-42c6-85a2-5c001873a4a4">


Where:
- `t` represents the term,
- `d` represents a document,
- and `D` represents the set of all documents.

The resulting TF-IDF score reflects the importance or relevance of a word to a document in a collection. Words that are frequent in a document but not common across documents (high TF-IDF score) are considered significant.

### Usage in SVM

Once the text data is transformed into TF-IDF vectors, these vectors serve as input features for machine learning models, such as the Support Vector Machine (SVM). SVM then uses these features to classify documents into different categories based on the patterns learned during the training phase. The high-dimensional nature of TF-IDF vectors pairs well with SVM's strength in handling high-dimensional spaces, making this combination particularly effective for text classification tasks in NLP.

### Word Embeddings:

Methods like Word2Vec or GloVe convert words into vectors, capturing semantic relationships.

## 4. Modeling

This project employs machine learning models to classify the sentiment of financial news articles. Two main models are used:

### SVM (Support Vector Machine)
SVM is a robust classifier known for its effectiveness in high-dimensional spaces, such as those encountered in text classification. It works by finding the hyperplane that best separates different classes in the feature space. For our project, SVM is used to classify news articles into positive, negative, or neutral sentiments based on their TF-IDF vector representations. The choice of SVM is driven by its strong performance in similar text classification tasks and its ability to handle the complexity of natural language data.

### Sentiment Analyzer Package
In addition to SVM, we compare our results with a sentiment analyzer package, which offers a pre-trained model for sentiment analysis. This comparison aims to benchmark our custom SVM model against a standard tool in the field. The sentiment analyzer provides a quick, general assessment of text sentiment, serving as a useful reference point for evaluating the effectiveness and accuracy of our SVM model in capturing the nuances of financial news sentiment.

## 5. Sentiment Analysis and Evaluation

After preprocessing and feature extraction, we apply Support Vector Machine (SVM) models to perform sentiment analysis on financial news articles. The goal is to classify each article into one of three sentiment categories: positive, negative, or neutral.

### Sentiment Analysis Using SVM

The SVM model is particularly suited for this task due to its effectiveness in handling high-dimensional data, such as text represented by TF-IDF vectors. We trained the SVM model on a labeled dataset of financial news articles, where each article was assigned a sentiment label based on its content.

### Evaluation Metrics

To assess the performance of our sentiment analysis models, we use several evaluation metrics:

#### Accuracy
Accuracy is the measure of all correct predictions (true positives, true negatives and true neutrals) made by the model over the total number of cases examined.

#### Precision (Positive Predictive Value)
Precision measures the proportion of true positive results in the total number of positive predictions made.

#### Recall (Sensitivity)
Recall measures the proportion of true positive results in the total number of actual positives.

#### F1 Score
The F1 Score is the harmonic mean of precision and recall, providing a balance between the two metrics for cases where one might be more important than the other.

<img width="273" alt="Screenshot 2024-03-20 at 00 24 01" src="https://github.com/alperencanbey/Financial-News-Sentiment-Analysis/assets/88103433/98ca062a-a283-4224-8ec0-88d447b640f7">

<img width="570" alt="Screenshot 2024-03-20 at 00 22 51" src="https://github.com/alperencanbey/Financial-News-Sentiment-Analysis/assets/88103433/8d030ad2-f8c9-47dd-af5c-d926a49d50ea">



# Variable Construction Formulas and Explanations

Understanding the impact of financial news sentiment on stock market movements involves constructing several key variables. Below, we detail these variables along with their mathematical definitions and explanations.

## Net Sentiment Score

The Net Sentiment Score represents the difference between the number of positive and negative articles, giving an overall sentiment direction for a given time period.

"Net Sentiment Score = (Number of Positive Articles) - (Number of Negative Articles)"

![Net Sentiment Score Equation](URL_TO_NET_SENTIMENT_SCORE_IMAGE)

This score helps identify whether the overall sentiment for the period is more positive or negative, which can be correlated with market movements.

## Sentiment Balance

The Sentiment Balance measures the ratio of positive articles to the total number of sentiment-bearing articles, indicating the prevalence of positive sentiment.

"Sentiment Balance = Number of Positive Articles / (Number of Positive Articles + Number of Negative Articles)"

![Sentiment Balance Equation](URL_TO_SENTIMENT_BALANCE_IMAGE)

A higher Sentiment Balance suggests a predominance of positive sentiment, which may influence market optimism.

## Sentiment Diversity

Sentiment Diversity uses Shannon entropy to quantify the uniformity or diversity of sentiment in news articles, considering all sentiment categories.

"Sentiment Diversity = -sum(p_i * log(p_i)) over all i in {positive, negative, neutral}"

![Sentiment Diversity Equation](URL_TO_SENTIMENT_DIVERSITY_IMAGE)

This variable captures how varied the sentiment is, with higher values indicating a broader range of sentiments being expressed.

## Sentiment Momentum

Sentiment Momentum captures the change in sentiment over time, highlighting shifts in the sentiment landscape from one day to the next.

"Sentiment Momentum on day t = Net Sentiment Score on day t - Net Sentiment Score on day t-1"

![Sentiment Momentum Equation](URL_TO_SENTIMENT_MOMENTUM_IMAGE)

By measuring day-to-day changes, Sentiment Momentum can reveal trends in sentiment that may precede market reactions.


## 8. Integration and Automation

The entire workflow from data collection to sentiment analysis is automated, enabling the system to periodically process new data and update its predictions.

## Comparison: SVM vs. Sentiment Analyzer

- **SVM Model:** Custom-trained on our dataset, offering tailored analysis sensitive to the specific language and sentiment of financial news. The SVM model, especially with negation handling and TF-IDF features, provides nuanced insights into sentiment.
- **Sentiment Analyzer Package:** Offers a general approach to sentiment analysis, useful for broad applications but potentially less sensitive to the specific nuances of financial news sentiment.

Our comparative analysis revealed that while the sentiment analyzer package provides a quick and general assessment of sentiment, the SVM model, with its custom training and feature engineering (including negation handling), delivers more accurate and contextually relevant insights for financial news.

## Conclusion

This project underscores the effectiveness of NLP and machine learning in analyzing financial news sentiment and its impact on stock markets. Through careful data preparation, feature engineering, and model evaluation, we've developed a robust system for predictive analysis, tailored to the complexities of financial news.
