Steps to run the code:

1. If you want to test data collection, run "python stream_twitter_emoticons.py"

2. If you want to run our code of sentiment Analysis using FCM - run "python sentimentAnalysis.py". Our output is in file - coding_output.txt

Using preprocessed data itself takes 23 minutes, so we are commenting the preprocessing part, and getting positive and negative sentiment scores from sentimentPosScore.txt and sentimentNegScore.txt


3. If you want to run skfuzzy version Fuzzy C-Means, run "python skfuzzy_cmeans.py". Our output can be found in file fuzzy_output.txt.

4. If you want to run hard K-means or K-means algorithm, run "python kmeans.py". Our output can be found in file kmeans_output.txt. Empty lines means no emoticons

5. During the preprocessing stage in sentimentAnalysis.py, emoticons are stores in emoticons.txt, hashtags are stored in hashTags.txt. Empty hashtags means no hashtag for that line, and the remaining preprocessed data can be found in preprocessed.txt. We 've shuffled the lines a lot so its best you run the code and check for yourself.

6. Plots can be found in plots folder. Using centroids value from first lines of coding_output.txt, fuzzy_output.txt and kmeans_output.txt, we get plots using plot_1d.py. Run it as "python plot_1d.py" 
