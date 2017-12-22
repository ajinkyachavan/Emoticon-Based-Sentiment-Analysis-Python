
#!/usr/bin/python
# -*- coding: UTF-8 -*-

#----- Use python 2.7-------


import skfuzzy as fuzz
import matplotlib.pyplot as plt

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import nltk
from emoji import UNICODE_EMOJI
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
np.set_printoptions(threshold=np.nan)

from  sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

#time for execution
import timeit
start_time = timeit.default_timer()


from random import  shuffle
import math

targetEmoticons = {1: "happy", 2: "love", 3: "playful", 4: "sad", 5: "angry", 6: "confused"}


## get positive and negative tweets from sentimentPosScore.txt
## and sentimentNegScore.txt which contains sentiment scores for positive and negative tweets
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


### add positive and negative scores to sentimentScore array and shuffle the array to avoid ordered data
sentimentScore = []

sentimentScore.extend(pos_tweets.flatten())
sentimentScore.extend(neg_tweets.flatten())
shuffle(sentimentScore)
sentimentScore = np.array(sentimentScore)
alldata = sentimentScore.reshape(-1, 1)


#### We are using FuzzyKMeans from skfuzzy because fuzzyCMeans of skfuzzy needs 2 dimensional data
#### and we have one-dimensional data.
####http://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

#### Fuzzy KMeans
#print('FUZZY_KMEANS')

fuzzy_kmeans = FuzzyKMeans(k=6, m=2)
fuzzy_kmeans.fit(alldata)
print np.sort((np.array(fuzzy_kmeans.cluster_centers_).flatten()))
# print(fuzzy_kmeans.cluster_centers_)
# print (kmeans.labels_)
# print (kmeans.cluster_centers_)
# print (kmeans.n_clusters)


### contains predicted centroids for all 6 emoticons, will be printed in fuzzy_output.txt
my_cluster_centers = np.array(fuzzy_kmeans.cluster_centers_).flatten().tolist()
my_cluster_centers = sorted(my_cluster_centers)
negative_centers = np.array(my_cluster_centers[:3]).flatten()
positive_centers = np.array(my_cluster_centers[3:]).flatten()
#print positive_centers, negative_centers


## cluster all data into 6 groups depending on which centroid its closest to
## e.g. 0.21 will be closest to centroid 0.22 as opposed to 0.08
finalClusters = []

def getClusters(sentiCenter, senti_tweets):
    cluster1 = []
    cluster2 = []
    cluster3 = []

    for i in range(len(senti_tweets)):

        dist1 = math.fabs(sentiCenter[0] - senti_tweets[i])
        dist2 = math.fabs(sentiCenter[1] - senti_tweets[i])
        dist3 = math.fabs(sentiCenter[2] - senti_tweets[i])

        # + str(emotweetIDs[i])+","
        if (dist1 > dist2):
            if (dist2 > dist3):
                # cluster3.append(str(emotweets[i])+","+ str(targets[emotweetIDs[i]]))
                cluster3.append(senti_tweets[i])
            else:
                cluster2.append(senti_tweets[i])
        else:
            if (dist1 > dist3):
                cluster3.append(senti_tweets[i])
            else:
                cluster1.append(senti_tweets[i])

    finalClusters.append(cluster1)
    finalClusters.append(cluster2)
    finalClusters.append(cluster3)

### get Clusters for positive tweets
finalClustersDictIdx = 1
getClusters(positive_centers, pos_tweets)


### get Clusters for negative tweets
finalClustersDictIdx = 4
getClusters(negative_centers, neg_tweets)



### combine positive and negative clusters and assign index of the predicted value to finalClustersIdx
### This will be the predicted target
finalClustersDict = {1:"", 2:"", 3:"", 4:"", 5:"", 6:""}
finalClustersIdx = []

k = 0
for clusters in finalClusters:
    finalClustersIdx.append([])

    for cluster in clusters:
        myidx = my_cluster_centers.index(min(my_cluster_centers, key=lambda x: abs(x - cluster)))+1
        #print cluster, my_cluster_centers,  myidx
        finalClustersIdx[k].append(myidx)
    k += 1


### get max value in each cluster. This will be the actual target
indices = []
for cluster in finalClustersIdx:
    indices.append(max(cluster,key=cluster.count))

#print indices
accuracy = [0]*len(my_cluster_centers)

k = 0
for cluster in finalClustersIdx:
    for row in cluster:
        if(row == indices[k]):
            accuracy[k] += 1
    k += 1

for row in range(len(accuracy)):
    #print  accuracy[row], len((finalClusters[row]))
    accuracy[row] = accuracy[row]/float(len(finalClusters[row]))*100


#print "Accuracy for individual emotions", accuracy
print "Average accuracy", sum(accuracy)/len(accuracy)