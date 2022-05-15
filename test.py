from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import re
from itertools import islice
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

catchy_words = ['BREAKING', 'NOW', 'LIVE', 'VIDEO', '#world', '#news',
                'MUST READ!', '#TopNews', '#TopVideo', 'politics', 'Terrorist', '#politics' 'More:' ]

excel_file = 'train.xlsx'
tweets = pd.read_excel(excel_file)
tweets_test=pd.read_excel("test.xlsx")

def prepare_data(data):
    data.dropna(subset = ["actions"], inplace=True)
    data.dropna(subset = ["location"], inplace=True)

    data['has_url'] = data['Tweet'].str.contains("http")
    data['contains_hashtag'] = data['Tweet'].str.contains("#")
    data['contains_catchy_words'] = data['Tweet'].str.contains('|'.join(catchy_words), flags=re.IGNORECASE, regex=True)
    data['contains_tweet_url'] = data['Tweet'].str.contains('https://t.co')

prepare_data(tweets)
prepare_data(tweets_test)

print(tweets.describe(exclude=[np.number]))


X_train, X_valid, y_train, y_valid = train_test_split(tweets["Tweet"], tweets["Type"], test_size=0.2)
print("Training Data: {}, Validation: {}".format(len(X_train), len(X_valid)))

data_v = CountVectorizer(max_features=5000, binary=True, stop_words="english")
data_v.fit(X_train)
X_train_v = data_v.transform(X_train)
X_valid_v = data_v.transform(X_valid)
data_v.vocabulary_
list(islice(data_v.vocabulary_.items(), 20))
model= LogisticRegression(C=0.3)
nieglupie =  model.fit(X_train_v, y_train)

print("Training Acc: {:.4f}".format(model.score(X_train_v, y_train)))
print("Validation Acc: {:.4f}".format(model.score(X_valid_v, y_valid)))

tweets_test.head()
ytest = np.array(y_valid)

print(classification_report(ytest, model.predict(X_valid_v)))
cm = confusion_matrix(ytest, model.predict(X_valid_v))

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels = model.classes_)
    # disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

disp.plot()
plt.savefig('foo.png')
