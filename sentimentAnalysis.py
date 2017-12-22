#!/usr/bin/python
# -*- coding: UTF-8 -*-

#----- Use python 2.7-------

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#time for execution
import timeit
start_time = timeit.default_timer()

from nltk.corpus import sentiwordnet as swn
import random
from sklearn import svm
import nltk
import re
from emoji import UNICODE_EMOJI
from nltk.stem.snowball import SnowballStemmer
import os
import re
import math
import numpy as np

#------------------- files read ---------------------------------------

pos = open('positive-words.txt')
neg = open('negative-words.txt')



unclean_file = open('data/final_tweets.csv')



#------------------- nltk variables -----------------------------------

words = list(set(w.lower() for w in nltk.corpus.words.words()))

stopWords = list(set(w.lower() for w in nltk.corpus.stopwords.words()))


stemmer = SnowballStemmer("english", ignore_stopwords=True)


posTweets = []
negTweets = []

posTweetIDs = []
negTweetIDs = []

#------------------- variable declaration ----------------------------------------

posWords = []
negWords = []


finalClustersDict = {1:"", 2:"", 3:"", 4:"", 5:"", 6:""}
finalClusters = []

#------------------- file data to lists -------------------------------------------

for line in pos:
    posWords.append(line.strip('\n').strip())

for line in neg:
    negWords.append(line.strip('\n').strip())

#-----------------recognize emojis--------------------------------------------------


def is_emoji(s):
    count = 0
    for emoji in UNICODE_EMOJI:
        count += s.count(emoji)
        if count > 1:
            return False
    return bool(count)


# --------- emoji sentiment rank from http://kt.ijs.si/data/Emoji_sentiment_ranking/ ---------------------

emoji_SentimentScores = {}

#happy, angry, love, sad, playful, confused
emoji_SentimentScores["\xF0\x9F\x98\x82"] = 0.221 #0.221*2
emoji_SentimentScores["\xF0\x9F\x98\xA1"] = -0.173 #-0.173
emoji_SentimentScores["\xe2\x9d\xa4"] = 0.746 #0.746*2
emoji_SentimentScores["\xF0\x9F\x98\xAD"] = -0.093 #-0.093*2
emoji_SentimentScores["\xF0\x9F\x98\x9C"] = 0.445 #0.445*2
emoji_SentimentScores["\xf0\x9f\x98\x95"] = -0.397 #0.397*2

#if a positive or negative word is encountered, the sentimentScore will be changed by this value
#0.124833 = average of all emoji scores
averageChangeInSentiment = sum(emoji_SentimentScores.values())/len(emoji_SentimentScores.values())



#--------- declare targets --------------------------

targetEmoticons = {1: "happy", 2: "love", 3: "playful", 4: "sad", 5: "angry", 6: "confused"}

#------------------------------ remove stopwords ---------------------------------------------------------------------




tweets = []


#happy, angry, love, sad, playful, confused
for row in unclean_file.readlines():
    #remove usernames
    row = ' '.join(re.sub("(@[A-Za-z0-9_]+)", "", row).split())

    #remove stopWords
    wordList = row.split()
    for word in wordList:
        if word in stopWords or len(word) == 1:
            row = row.replace(word, "")
            #print (word, row)
    try:
        tweets.append(row)
    except:
        pass

#happy, love, playful, sad, angry, confused
targets = [0]*len(tweets)

for target in range(len(tweets)):
    if("\xF0\x9F\x98\x82" in tweets[target]):
        targets[target] = 1
    elif("\xF0\x9F\x98\xA1" in tweets[target]):
        targets[target] = 5
    elif("\xe2\x9d\xa4" in tweets[target]):
        targets[target] = 2
    elif("\xF0\x9F\x98\xAD" in tweets[target]):
        targets[target] = 4
    elif("\xF0\x9F\x98\x9C" in tweets[target]):
        targets[target] = 3
    elif("\xf0\x9f\x98\x95" in tweets[target]):
        targets[target] = 6

####------------------- uncomment from line 145 to 367 to perform data preprocessing and get sentiment Score
####------------------- else the positive sentiment scores are stored in sentimentPosScore.txt
####------------------- and negative sentiment in sentimentNegScore.txt



# #------------------ declare vars to store #, emojis, POS tags, sentiment Score for each tweet  -------------------------------
#
# hashtags = [""]*len(tweets)
# emojis = [""]*len(tweets)
# POStags = [""]*len(tweets)
#
# sentimentScore = [0]*len(tweets)
# sentiWord = [0] * len(tweets) #collector of sentiment Scores
#
# #--------------------------- assign #, emojis and store POS tags in POStags[] ------------------------------------
#
# #tweets = tweets[:10]
#
# idx = 0 #tweet counter
#
# for sentence in tweets:
#     hashtags[idx] = []
#     emojis[idx] = []
#
#     splitSentence = sentence.split()
#     #print (splitSentence, idx)
#
#     for word in splitSentence:
#         #print (word, "#" in word, is_emoji(word))
#         if "#" == word[0]:
#             hashtags[idx].append(word[1:])
#         elif(is_emoji(word)):
#             emojis[idx].append(word)
#
#
#     #remove emoji data, usernames to apply POS
#     tweets[idx] = ' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweets[idx]).split())
#     sentence = ' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence).split())
#     txt = nltk.word_tokenize(sentence)
#
#     try:
#         #txt = txt.decode('cp1252').encode('utf-8')
#         POStags[idx] = nltk.pos_tag(txt)
#     except:
#         pass
#     idx += 1
#
#
# #------------------------------- get sentimentScore of emojis -------------------------------
# #
# for i in range(len(emojis)):
#     if(len(emojis[i]) == 0):
#         pass
#     else:
#         #match sentiword with hashtag data
#         for j in emojis[i]:
#             if (j in emoji_SentimentScores.keys()):
#                 sentimentScore[i] += emoji_SentimentScores[j]
#
#
# #------------------------------- logic to convert hashtags like ThisIsCamelCasing to [This, is, camel, casing]  ----------------------
#
# def f1(w,c) : return list(zip(* filter(lambda (x,y): x == c, zip(w, range(len(w)))  ))[1])
#
# def getCamelCaseList(j):
#
#     uppers = list(set([j.index(l) for l in j if l.isupper()]))
#
#     indices = []
#
#     for i in range(len(uppers)):
#         indices.append(f1(j, j[uppers[i]]))
#
#     indices.append([len(j)])
#
#     indices = [item for sublist in indices for item in sublist]
#
#     flat_list = []
#
#     if (indices != []):
#         for k in range(len(indices)-1):
#             #print (j, j[indices[k]:indices[k + 1]])
#             flat_list.append(str(j[indices[k]:indices[k + 1]]).lower())
#
#     return  flat_list
#
# #-------------------------------- get SentimentScore of hashtags ------------------------------
#
# for i in range(len(hashtags)):
#     val = 0
#
#     if (len(hashtags[i]) == 0):
#         pass
#     else:
#         for hashWord in hashtags[i]:
#
#             newj = getCamelCaseList(hashWord)
#
#             if(newj != [] or hashWord!=""):
#                 if(len(newj)>1):
#                     #print (newj)
#                     for j in newj:
#                         if j in posWords:
#                             #print "hash pos many ", hashWord
#                             sentimentScore[i] += averageChangeInSentiment*2
#                         elif j in negWords:
#                             #print "hash neg many ", hashWord
#                             sentimentScore[i] -= averageChangeInSentiment*2
#                 else:
#                     #print (hashWord)
#                     if hashWord in posWords:
#                         #print "hash pos ",hashWord
#                         sentimentScore[i] += averageChangeInSentiment*2
#                     elif hashWord in negWords:
#                         #print "hash neg ",hashWord
#                         sentimentScore[i] -= averageChangeInSentiment*2
#
#
# #---------------------------------- POS tagging -----------------------------
#
# class Splitter(object):
#     def __init__(self):
#         self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
#         self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
#
#     def split(self, text):
#
#         sentences = self.nltk_splitter.tokenize(text)
#         tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
#         return tokenized_sentences
#
#
# class POSTagger(object):
#     def __init__(self):
#         pass
#
#     def pos_tag(self, sentences):
#
#         pos = [nltk.pos_tag(sentence) for sentence in sentences]
#         # adapt format
#         pos = [[postag for (word, postag) in sentence] for sentence in pos] #(word, [postag])
#         return pos
#
# for i in range(len(tweets)):
#
#     splitter = Splitter()
#     postagger = POSTagger()
#
#     splitted_sentences = splitter.split(tweets[i])
#
#     pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
#
#     for j in range(len(pos_tagged_sentences)):
#         for k in range(len(pos_tagged_sentences[j])):
#             #print pos_tagged_sentences[j][k], pos_tagged_sentences[j][k] in ['NN', 'NNS', 'NNP'], \
#                 # pos_tagged_sentences[j][k] in ['JJ', 'JJR', 'JJS'], \
#                 # pos_tagged_sentences[j][k] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], \
#                 # splitted_sentences[j][k], splitted_sentences[j][k] in posWords, splitted_sentences[j][k] in negWords, \
#                 # sentimentScore[i]
#
#             try:
#                 if(pos_tagged_sentences[j][k] in ['NN', 'NNS', 'NNP']):
#                     #print "noun ahe"
#                     if(splitted_sentences[j][k] in posWords):
#                         if(k>0 and pos_tagged_sentences[j][k-1] in ['JJ', 'JJR', 'JJS']):
#                             #print "pos k > 0"
#                             sentimentScore[i] += averageChangeInSentiment*2
#                         else:
#                             #print "fakt noun pos"
#                             sentimentScore[i] += averageChangeInSentiment
#                     elif(splitted_sentences[j][k] in negWords):
#                         if(k>0 and pos_tagged_sentences[j][k-1] in ['JJ', 'JJR', 'JJS']):
#                             #print "neg k>0"
#                             sentimentScore[i] -= averageChangeInSentiment*2
#                         else:
#                             #print "fak noun neg"
#                             sentimentScore[i] -= averageChangeInSentiment
#                 elif(pos_tagged_sentences[j][k] in ['JJ', 'JJR', 'JJS']):
#                     if splitted_sentences[j][k] in posWords:
#                         sentimentScore[i] += averageChangeInSentiment*2
#                     elif splitted_sentences[j][k] in negWords:
#                         sentimentScore[i] -= averageChangeInSentiment*2
#                 elif(pos_tagged_sentences[j][k] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
#                     stemmed_verb = stemmer.stem(splitted_sentences[j][k])
#                     #print " stem verb ",stemmed_verb, stemmed_verb in posWords, stemmed_verb in negWords
#                     if stemmed_verb in posWords:
#                         sentimentScore[i] += averageChangeInSentiment*2
#                     elif stemmed_verb in negWords:
#                         sentimentScore[i] -= averageChangeInSentiment*2
#             except:
#                 pass
#
#
# #------------------ print SentimentScore ----------
#
# # for score in range(len(sentimentScore)):
# #     print (tweets[score], sentimentScore[score], targets[score])
#
# for score in range(len(sentimentScore)):
#     if(sentimentScore[score] > 0):
#         posTweets.append(sentimentScore[score])
#         posTweetIDs.append(score)
#     elif(sentimentScore[score] < 0):
#         negTweets.append(sentimentScore[score])
#         negTweetIDs.append(score)
#
# print ("--------------tweets-------------")
# #print (len(sentimentScore))
# #out = [x for x in sentimentScore if x>0]
# #print ("out", len(out), out)
#
# sentimentScorePositiveFile = open('sentimentPosScore.txt', 'w')
# for i in range(len(posTweets)):
#     if(i != len(posTweets)-1):
#         sentimentScorePositiveFile.write(str(posTweetIDs[i])+":"+str(posTweets[i])+",")
#     else:
#         sentimentScorePositiveFile.write(str(posTweetIDs[i])+":"+str(posTweets[i]))
#
# sentimentScoreNegativeFile = open('sentimentNegScore.txt', 'w')
# for i in range(len(negTweets)):
#     if (i != len(negTweets) - 1):
#         sentimentScoreNegativeFile.write(str(negTweetIDs[i])+":"+str(negTweets[i]) + ",")
#     else:
#         sentimentScoreNegativeFile.write(str(negTweetIDs[i])+":"+str(negTweets[i]))

#######------------------- files  sentimentPosScore.txt & sentimentNegScore.txt contain
#######------------------- sentiment scores processed from line 145 to line 367
###-- comment from line 372 to 389 after uncommenting above lines to get sentiment scores again.

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




#------------------ fuzzy C Means ---------------------


m = 2.0 #membership_constant

def getNumerator(arrayOfMemArray, emotweets):

    mem = [0]*3

    for i in range(len(emotweets)):
        for j in range(3):
            mem[j] += math.pow(arrayOfMemArray[i][j], m)*emotweets[i]
    return mem


def getDenominator(arrayOfMemArray, emotweets):
    mem = [0]*3

    for i in range(len(emotweets)):
        for j in range(3):
            mem[j] += math.pow(arrayOfMemArray[i][j], m)

    return mem


def getMatrixDifference(arrayOfMemArray, emotweets):
    diff1 = 0; diff2 = 0

    for i in range(len(emotweets)):
        if i == 0:
            diff1 = math.fabs(arrayOfMemArray[i][0] - arrayOfMemArray[i][1])
            diff2 = math.fabs(arrayOfMemArray[i][1] - arrayOfMemArray[i][2])
        new_diff1 = math.fabs(arrayOfMemArray[i][0] - arrayOfMemArray[i][1])
        new_diff2 = math.fabs(arrayOfMemArray[i][1] - arrayOfMemArray[i][2])

        if(new_diff1 < diff1):
            diff1 = new_diff1
        if(new_diff2 < diff2):
            diff2 = new_diff2

        if(diff1 > 0.15 and diff2 > 0.15 and i<len(emotweets)-1):
            return True
    return  False

finalCentroids = []
def fuzzyCMeans(emotweets, emotweetIDs, targets, centroids, finalClustersDictIdx):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(len(emotweets)) :

        arrayOfMemArray = []
        for score in emotweets:
            score = float(score)
            membershipArray = []

            for centroid_j in centroids:
                membership = 0
                for centroid_k in centroids:
                    membership += math.pow( math.fabs(score-centroid_j) / math.fabs( (score+0.001)-centroid_k), 2/(m-1) )

                membership = 1/(membership+0.001)

                membershipArray.append(membership)

            arrayOfMemArray.append(membershipArray)

        numerator = np.array(getNumerator(arrayOfMemArray, emotweets))
        denominator = np.array(getDenominator(arrayOfMemArray, emotweets))

        centroids = numerator/denominator

        boolVal = getMatrixDifference(arrayOfMemArray, emotweets)

        if boolVal == True or i==len(emotweets)-1:

            for i in range(len(emotweets)):

                dist1 = math.fabs(centroids[0] - emotweets[i])
                dist2 = math.fabs(centroids[1] - emotweets[i])
                dist3 = math.fabs(centroids[2] - emotweets[i])

                #+ str(emotweetIDs[i])+","
                if(dist1 > dist2):
                    if(dist2 > dist3):
                        #cluster3.append(str(emotweets[i])+","+ str(targets[emotweetIDs[i]]))
                        cluster3.append(int(targets[emotweetIDs[i]]))
                    else:
                        cluster2.append(int(targets[emotweetIDs[i]]))
                else:
                    if(dist1 > dist3):
                        cluster3.append(int(targets[emotweetIDs[i]]))
                    else:
                        cluster1.append(int(targets[emotweetIDs[i]]))

    finalClusters.append(cluster1)
    finalClusters.append(cluster2)
    finalClusters.append(cluster3)

    for centroid in centroids:
        finalCentroids.append(centroid)


#####---- if you are uncommenting line 145 to 367 and commenting lines 372 to 389, uncomment 502 and 508
#####---- & comment 505 & 511. Currently we are using data from sentimentPosScore and NegScore
#####---- so we are using lines 505 and 511
#####---- else use lines 502 and 508

finalClustersDictIdx = 1
#fuzzyCMeans(posTweets, posTweetIDs, targets, [0, 1, 3], finalClustersDictIdx)

#for test version with senti files
fuzzyCMeans(pos_tweets, pos_tweets_id, targets, [0, 0.5, 1], finalClustersDictIdx)

finalClustersDictIdx = 4
#fuzzyCMeans(negTweets, negTweetIDs, targets, [0, -1, -3], finalClustersDictIdx) #full version

#test
fuzzyCMeans(neg_tweets, neg_tweets_id, targets, [0, -0.25, -0.5], finalClustersDictIdx)

print np.array(finalCentroids).flatten()

finalClustersDict = {1:"", 2:"", 3:"", 4:"", 5:"", 6:""}
finalClustersIdx = []

k = 0
for clusters in finalClusters:
    finalClustersIdx.append([])

    for cluster in clusters:
        myidx = finalCentroids.index(min(finalCentroids, key=lambda x: abs(x - cluster)))+1
        #print cluster, centroids,  myidx
        finalClustersIdx[k].append(myidx)
    k += 1

indices = []
for cluster in finalClustersIdx:
    indices.append(max(cluster,key=cluster.count))

#print indices
accuracy = [0]*len(finalCentroids)

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

elapsed = timeit.default_timer() - start_time

print("Time elapsed is ",elapsed)