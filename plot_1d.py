import numpy as np
import matplotlib.pyplot as pp
from random import  shuffle


val = 0. # this is the value where you want the data to appear on the y-axis.

pos_tweets = []
pos_tweets_id = []
neg_tweets = []
neg_tweets_id = []


pos_tweets_file = open('sentimentPosScore.txt')
neg_tweets_file = open('sentimentNegScore.txt')

read_pos = pos_tweets_file.read().split(",")
for row in read_pos:
    pos_tweets_id.append(int(row.split(":")[0]))
    pos_tweets.append(float(row.split(":")[1]))

read_neg = neg_tweets_file.read().split(",")
for row in read_neg:
    neg_tweets_id.append(int(row.split(":")[0]))
    neg_tweets.append(float(row.split(":")[1]))

pos_tweets = np.array(pos_tweets)
neg_tweets = np.array(neg_tweets)

shuffle(pos_tweets)
shuffle(neg_tweets)

## pos_vs_neg_plot
#pp.plot(pos_tweets, 'r--', neg_tweets, 'g--')

## cluster with centroids plots for our code
#senti_read = open('coding_output.txt')

## cluster with centroids plots for skfuzzy
#senti_read = open('fuzzy_output.txt')

## cluster with centroids plots for kmeans
senti_read = open('kmeans_output.txt')


line = senti_read.readlines()[0]
line = line[1:len(line)-2]
centroids = []
for word in line.split():
    centroids.append(float(word))


max_len = len(pos_tweets)
if(len(pos_tweets) < len(neg_tweets)):
    max_len = len(neg_tweets)

pp.scatter(pos_tweets,  np.zeros_like(pos_tweets) + val, c='red')
pp.scatter(neg_tweets,  np.zeros_like(neg_tweets) + val, c='green')
pp.scatter(centroids,  np.zeros_like(centroids) + val, c='black')

#pp.scatter(max_len, neg_tweets, c='green')
pp.show()