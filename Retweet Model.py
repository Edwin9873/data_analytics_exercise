import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse
import eli5

#%% Data Loading
tweets = pd.read_excel (r'D:\Dropbox\Collinson\tweets.xlsx')
tweets.dtypes
tweets['TweetBodyLength'] = tweets.TweetBody.str.len()
tweets['TweetHashtagsCount'] = tweets.TweetHashtags.str.count(",") + 1
tweets['TweetRetweetCount'] = tweets['TweetRetweetCount'].astype('int')

#%%
# Data Exploratory

for column in tweets.select_dtypes(include=["object"]).columns:
    display(pd.crosstab(index=tweets[column], columns="% observations", normalize="columns"))
pd.plotting.scatter_matrix(tweets[['TweetBodyLength', 'TweetHashtagsCount', 'TweetRetweetCount', 'TweetFavoritesCount', 'UserFollowersCount', 'UserFriendsCount', 'UserListedCount', 'UserTweetCount']], alpha=0.2)
pd.plotting.scatter_matrix(tweets[['TweetBodyLength', 'TweetHashtagsCount', 'TweetRetweetCount', 'TweetFavoritesCount']], alpha=0.2)

#ax = tweets['TweetRetweetCount'].plot.hist(bins=12, alpha=0.5)

## Quick df facts
#df_size = tweets.count()
##num_unique_tweets = tweets.TweetID.value_counts()
#num_unique_tweets = tweets.TweetID.nunique()
#num_unique_users = tweets.UserID.nunique()
#num_unique_users_reteweeted = tweets.TweetRetweetFlag.value_counts()
#user_tweet_count_df = tweets.groupby("UserID").nunique()

#%% By binning the data, a regression problem was converted into a classification problem which is easier to deal with when working with predictions
bins_cut = [0, 1, 3, 10, 100, 1000,5000]
labels_cut = [1,2,3,4,5,6]

tweets['TweetRetweetBin'] = pd.cut(tweets['TweetRetweetCount'], bins=bins_cut, labels=labels_cut )
tweets['TweetRetweetBin'] = tweets['TweetRetweetBin'].cat.add_categories(0).fillna(0)
tweets['TweetRetweetBin'].cat.categories = [0,1,2,3,4,5,6]

#%% Test and Train
training_data, testing_data = train_test_split(tweets,random_state = 2000)
text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)

X_train_text = text_transformer.fit_transform(training_data['TweetBody'])
X_test_text = text_transformer.transform(testing_data['TweetBody'])

X_train_text.shape, X_test_text.shape


logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

#%% Cross Validation
cv_results = cross_val_score(logit, X_train_text, training_data['TweetRetweetBin'], cv=skf, scoring='f1_micro')
cv_results, cv_results.mean()

#Test Data
logit.fit(X_train_text, training_data['TweetRetweetBin'])
eli5.show_weights(estimator=logit, 
                  feature_names= list(text_transformer.get_feature_names()),
                 top=(50, 5))

test_preds = logit.predict(X_test_text)
pd.DataFrame(test_preds, columns=['TweetRetweetBin'])

# Measure of Performance
test_result = testing_data['TweetRetweetBin'].reset_index()
test_result['Result'] = pd.DataFrame(test_preds, columns=['TweetRetweetBin'])

print('\nConfusion matrix\n',confusion_matrix(test_result['TweetRetweetBin'],test_result['Result']))
print(classification_report(test_result['TweetRetweetBin'],test_result['Result']))
