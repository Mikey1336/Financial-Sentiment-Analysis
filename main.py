from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import RocCurveDisplay, confusion_matrix, accuracy_score ,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import metrics
import numpy as np
import operator
import glob
import cv2
import yfinance as yf
import pandas as pd
import datetime
import math

nltk.download('vader_lexicon')
dat = pd.read_excel('twitonomy_elonmusk.xlsx')
dat['Negative'] = 0.0
dat['Neutral'] = 0.0
dat['Positive'] = 0.0
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

'''
for tw in musk_dat['Text']:
	words = []

	for word in tw.split(' '):
    	if word.startswith('@') and len(word) > 1:
        	word = '@user'
    
    	elif word.startswith('http'):
        	word = "http"
    words.append(word)

	tweet_proc = " ".join(words)
'''

for i in dat.iterrows():
	tw = i[1][3]
	words = []
	for word in tw.split(' '):
		if word.startswith('@') and len(word) > 1: word = '@user'
		elif word.startswith('http'): word = "http"
	words.append(word)

	tweet_proc = " ".join(words)
	#print(tweet_proc)
	encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
	output = model(**encoded_tweet)
	scores = output[0][0].detach().numpy()
	scores = softmax(scores)
	#print("tweet: "+ tw)
	for j in range(len(scores)):
		l = labels[j]
		s = scores[j]
		i[1][9] = scores[0]
		i[1][10] = scores[1]
		i[1][11] = scores[2]
		
	dat.to_excel('twitonomy_testsentiment.xlsx')

#Todo: normalize sentiment from range(-1,1) including neutral sentiment.
#if sentiment in range(-0.3,0.3), not a strong opinion
	
doge = yf.Ticker('DOGE-USD')
doge_dat = doge.history(period = "max")

tesla = yf.Ticker('TSLA')
tesla_dat = tesla.history(period = "Jan,Feb,Mar")

features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()

training_file = 'train.csv'
testing_file = 'test.csv'
positive_file = 'positive.txt'
negative_file = 'negative.txt'
neutral_file = 'neutral.txt'

def classify(processed_csv, test_file=True, **params):
    positive_words = utils.file_to_wordset(params.pop('positive_words'))
    negative_words = utils.file_to_wordset(params.pop('negative_words'))
    predictions = []
    with open(processed_csv, 'r') as csv:
        for line in csv:
            if test_file:
                tweet_id, tweet = line.strip().split(',')
            else:
                tweet_id, label, tweet = line.strip().split(',')
            pos_count, neg_count = 0, 0
            for word in tweet.split():
                if word in positive_words:
                    pos_count += 1
                elif word in negative_words:
                    neg_count += 1
            # print pos_count, neg_count
            prediction = 1 if pos_count >= neg_count else 0
            if test_file:
                predictions.append((tweet_id, prediction))
            else:
                predictions.append((tweet_id, int(label), prediction))
    return predictions

cleaned_tweets = pd.read_csv("twitonomy_testsentiment.csv")
vader = SentimentIntensityAnalyzer()
tweet_sentiment = pd.DataFrame()
tweet_sentiment['tweet'] = cleaned_tweets['text']
scores = cleaned_tweets['text'].apply(vader.polarity_scores).tolist()
scores_df = pd.DataFrame(scores)

tweet_sentiment["follower_count"] = cleaned_tweets["follower_count"]
tweet_sentiment["neg"] = scores_df["neg"]
tweet_sentiment['neu'] = scores_df['neu']
tweet_sentiment["pos"] = scores_df["pos"]
tweet_sentiment["compound"] = scores_df["compound"]
tweet_sentiment["time"] = cleaned_tweets['date'].values.astype('datetime64[ns]')
tweet_sentiment.to_csv(index=False, path_or_buf="tweet_data/hashtag_tweet_sentiment.csv")
weighed_tweets = Weigh_Tweets(tweet_sentiment)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)
'''
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
'''
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
bayes_graph = Y_pred = gaussian.predict(X_test) 

'''
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
dec_graph = Y_pred = decision_tree.predict(X_test) 

linear_svc = LinearSVC(max_iter=4000)
linear_svc.fit(X_train, y_train)
SVM_graph = Y_pred = linear_svc.predict(X_test)

'''

#TODO IMPLEMENT SLIDER





	