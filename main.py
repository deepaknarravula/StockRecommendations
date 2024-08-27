#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install update textblob')
import nltk

nltk.download('punkt')
get_ipython().system('pip install tweepy --upgrade')
get_ipython().system('pip install yfinance')
get_ipython().system('pip install yahoofinancials')

# In[9]:


import pandas as pd

# In[10]:


import requests

url = 'https://raw.githubusercontent.com/alvarobartt/twitter-stock-recommendation/master/companylist.csv'
res = requests.get(url, allow_redirects=True)
with open('companylist.csv', 'wb') as file:
    file.write(res.content)
companylist = pd.read_csv('companylist.csv')

# In[11]:


companylist

# In[12]:


consumer_key = 'yfBRroSOsbDLVF1dUxW1eHzUx'
consumer_secret = 'N5xDS7dPb0wA8iXp7cW8QMRZtbgVfDKSHrVBcCYuBtINGUHreI'
access_token = '1621302344867512320-1GWqLTAIRXhB6lAxh4anr7F1dskF0Y'
access_token_secret = 'n18scuwc5kWtr6QvRnmBZGIrcnhWLDkueiixs4PReTJt7'


# In[13]:


class Tweet(object):

    def __init__(self, content, polarity):
        self.content = content
        self.polarity = polarity


# In[14]:


import datetime as dt
import math

# import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tweepy
from matplotlib import style
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import yfinance as yf
from yahoofinancials import YahooFinancials

# In[15]:


style.use('ggplot')


# In[16]:


def check_stock_symbol(companies_file='companylist.csv'):
    df = pd.read_csv(companies_file, usecols=[0])
    # print(df)
    symbol = input('Enter a stock symbol: ').upper()
    found = False
    for index in range(len(df)):
        if df['Symbol'][index] == symbol:
            found = True
            break

    if not found:
        print(f"Stock symbol '{symbol}' not found.")
    return found, symbol


# In[17]:


def get_stock_data(symbol, from_date, to_date):
    data = yf.download(tickers=symbol, start=from_date, end=to_date)
    df = pd.DataFrame(data=data)
    # print(df)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HighLoad', 'Change', 'Volume']]
    return df


# In[18]:


def finding_the_best_model(df):
    forecast_column = 'Close'
    forecast_periods = int(math.ceil(0.1 * len(df)))
    df['Label'] = df[[forecast_column]].shift(-forecast_periods)

    feature_matrix = np.array(df.drop(['Label'], axis=1))
    feature_matrix = preprocessing.scale(feature_matrix)
    forecast_matrix = feature_matrix[-forecast_periods:]
    feature_matrix = feature_matrix[:-forecast_periods]

    df.dropna(inplace=True)
    target_vector = np.array(df['Label'])

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_vector, test_size=0.2)

    # Train and evaluate a linear regression model
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    linear_reg_pred = linear_reg_model.predict(X_test)
    linear_reg_mse = mean_squared_error(y_test, linear_reg_pred)

    # Train and evaluate a random forest regression model
    rf_reg_model = RandomForestRegressor()
    rf_reg_model.fit(X_train, y_train)
    rf_reg_pred = rf_reg_model.predict(X_test)
    rf_reg_mse = mean_squared_error(y_test, rf_reg_pred)

    # Train and evaluate a support vector regression model
    svm_reg_model = SVR()
    svm_reg_model.fit(X_train, y_train)
    svm_reg_pred = svm_reg_model.predict(X_test)
    svm_reg_mse = mean_squared_error(y_test, svm_reg_pred)

    # Train and evaluate a multilayer perceptron regression model
    mlp_reg_model = MLPRegressor()
    mlp_reg_model.fit(X_train, y_train)
    mlp_reg_pred = mlp_reg_model.predict(X_test)
    mlp_reg_mse = mean_squared_error(y_test, mlp_reg_pred)

    # Print the MSE scores for each model
    print('Linear Regression MSE:', linear_reg_mse)
    print('Random Forest Regression MSE:', rf_reg_mse)
    print('Support Vector Regression MSE:', svm_reg_mse)
    print('Multilayer Perceptron Regression MSE:', mlp_reg_mse)


# In[19]:


def stock_forecasting(df):
    forecast_column = 'Close'
    forecast_periods = int(math.ceil(0.1 * len(df)))
    df['Label'] = df[[forecast_column]].shift(-forecast_periods)

    feature_matrix = np.array(df.drop(['Label'], axis=1))
    feature_matrix = preprocessing.scale(feature_matrix)
    forecast_matrix = feature_matrix[-forecast_periods:]
    feature_matrix = feature_matrix[:-forecast_periods]

    df.dropna(inplace=True)
    target_vector = np.array(df['Label'])

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_vector, test_size=0.2)

    rf_reg_model = RandomForestRegressor()
    rf_reg_model.fit(X_train, y_train)
    accuracy = rf_reg_model.score(X_test, y_test)
    rf_reg_score = r2_score(y_test, rf_reg_model.predict(X_test))
    predictions = rf_reg_model.predict(forecast_matrix)

    df['Prediction'] = np.nan

    last_date = str(df.iloc[-1].name)
    last_date = last_date
    last_date = dt.datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")

    for prediction in predictions:
        last_date += dt.timedelta(days=1)
        df.loc[last_date.strftime("%Y-%m-%d")] = [np.nan for _ in range(len(df.columns) - 1)] + [prediction]

    return df, forecast_periods


# In[20]:


def forecast_plot(df):
    df['Close'].plot(color='black')
    df['Prediction'].plot(color='green')
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


# In[21]:


def retrieving_tweets_polarity(symbol):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Define a search term and the number of tweets to retrieve
    search_term = str(symbol)
    number_of_tweets = 100

    # Collect the tweets using the tweepy Cursor
    tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang='en', tweet_mode='extended').items(number_of_tweets)

    # Create a list to store the polarity scores of each tweet
    polarities = []

    # Loop through each tweet and calculate the polarity score
    for tweet in tweets:
        text = tweet.full_text
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        polarities.append(polarity)

    # Calculate the average polarity score
    avg_polarity = np.mean(polarities)
    # print(avg_polarity)
    return avg_polarity


# In[22]:


def recommending(df, forecast_out, global_polarity):
    if df.iloc[-forecast_out - 1]['Close'] < df.iloc[-1]['Prediction']:
        if global_polarity > 0:
            print(
                "According to the predictions and twitter sentiment analysis -> Investing in %s is a GREAT idea!" % str(
                    symbol))
        elif global_polarity < 0:
            print("According to the predictions and twitter sentiment analysis -> Investing in %s is a BAD idea!" % str(
                symbol))
    else:
        print("According to the predictions and twitter sentiment analysis -> Investing in %s is a BAD idea!" % str(
            symbol))


# In[23]:


if __name__ == "__main__":
    (flag, symbol) = check_stock_symbol('companylist.csv')
    if flag:
        actual_date = dt.date.today()
        past_date = actual_date - dt.timedelta(days=365 * 3)

        actual_date = actual_date.strftime("%Y-%m-%d")
        past_date = past_date.strftime("%Y-%m-%d")

        print("Retrieving Stock Data from introduced symbol...")
        dataframe = get_stock_data(symbol, past_date, actual_date)
        print("Finding the best model for the dataset")
        finding_the_best_model(dataframe)
        print("Forecasting stock DataFrame...")
        (dataframe, forecast_out) = stock_forecasting(dataframe)
        print("Plotting existing and forecasted values...")
        forecast_plot(dataframe)
        print("Retrieving %s related tweets polarity..." % symbol)
        polarity = retrieving_tweets_polarity(symbol)
        print(polarity)
        print("Generating recommendation based on prediction & polarity...")
        recommending(dataframe, forecast_out, polarity)

