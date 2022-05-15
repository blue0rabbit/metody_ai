from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np

catchy_words = ['BREAKING', 'NOW', 'LIVE', 'VIDEO', '#world', '#news']

excel_file = 'train.xlsx'
tweets = pd.read_excel(excel_file)


print(tweets)
spam = tweets[tweets["Type"] == "Spam"]

# spam_containing_tweet_url = spam[spam["Tweet"].str.contains("https://t.co/")]
# tweets_containning_tweet_url = tweets[tweets["Tweet"].str.contains("https://t.co/")]
#
# spam_containing_url = spam[spam["Tweet"].str.contains("http")]
# tweets_containning_url = tweets[tweets["Tweet"].str.contains("http")]
#
# spam_containing_catchy_words = spam[spam['Tweet'].str.contains('|'.join(catchy_words))]
# tweets_containing_catchy_words = tweets[tweets['Tweet'].str.contains('|'.join(catchy_words))]

# spam_location_containing_url = spam[(spam['location'] != None) & (spam['location'].str.contains('http'))]
tweets_location_containing_url = tweets[(tweets['actions']).isnull()]
 # & (tweets['location'].str.contains('http'))]

# a = ((spam_containing_tweet_url.size * 100)/tweets_containning_tweet_url.size)
# b = ((spam_containing_url.size * 100)/tweets_containning_url.size)
# c = ((spam_containing_catchy_words.size * 100)/tweets_containing_catchy_words.size)
d = ((tweets_location_containing_url.size * 100)/tweets.size)
#
# spam_types = [a, b, c, d]
#
print(d)
#twitt_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Reds').generate(spam)

# names = ['spam containing tweet url', 'spam containing url', 'spam containing catchy words' , 'spam location containing url']
# values = [a, b, c, d]
#
# plt.title("Spam percentage based on some conditions [%]")
# plt.bar(names, values, color=['pink'])
# plt.xticks(rotation = 45)
# plt.annotate(a,(0,a))
# plt.annotate(b,(1,b))
#
# plt.annotate(c,(2,c))
#
# plt.annotate(d,(3, d))
#
# plt.savefig('plot.png', bbox_inches="tight")
