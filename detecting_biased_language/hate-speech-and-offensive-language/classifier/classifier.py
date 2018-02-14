"""
This file contains code to

    (a) Load the pre-trained classifier and
    associated files.

    (b) Transform new input data into the
    correct format for the classifier.

    (c) Run the classifier on the transformed
    data and return results.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import heapq
import csv
from textstat.textstat import *


stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = VS()

stemmer = PorterStemmer()


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features_(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.

    This is modified to only include those features in the final
    model."""

    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet) #Get text only

    syllables = textstat.syllable_count(words) #count syllables in words
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['compound'],
                twitter_objs[2], twitter_objs[1],]
    #features = pandas.DataFrame(features)
    return features

def get_oth_features(tweets):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats=[]
    for t in tweets:
        feats.append(other_features_(t))
    return np.array(feats)


def transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer):
    """
    This function takes a list of tweets, along with used to
    transform the tweets into the format accepted by the model.

    Each tweet is decomposed into
    (a) An array of TF-IDF scores for a set of n-grams in the tweet.
    (b) An array of POS tag sequences in the tweet.
    (c) An array of features including sentiment, vocab, and readability.

    Returns a pandas dataframe where each row is the set of features
    for a tweet. The features are a subset selected using a Logistic
    Regression with L1-regularization on the training data.

    """
    tf_array = tf_vectorizer.fit_transform(tweets).toarray()
    tfidf_array = tf_array*idf_vector
    print "Built TF-IDF array"

    pos_tags = get_pos_tags(tweets)
    pos_array = pos_vectorizer.fit_transform(pos_tags).toarray()
    print "Built POS array"

    oth_array = get_oth_features(tweets)
    print "Built other feature array"

    M = np.concatenate([tfidf_array, pos_array, oth_array],axis=1)
    return pd.DataFrame(M)

def predictions(X, model):
    """
    This function calls the predict function on
    the trained model to generated a predicted y
    value for each observation.
    """
    y_preds = model.predict(X)
    return y_preds

def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate speech"
    elif class_label == 1:
        return "Offensive language"
    elif class_label == 2:
        return "Neither"
    else:
        return "No label"


def get_list_from_file(file_name):
    with open(file_name, "r") as f1:
        l = f1.read().lower().split('\n')
    return l

def transform_twitter(twitter, utterance_delimiter='</s>', cornell=False):
    dialogues = get_list_from_file(twitter)
    tweets = []
    for dialogue in dialogues:
        if utterance_delimiter is None:
            logs = [dialogue]
        else:
            logs = dialogue.split(utterance_delimiter)
        if cornell:
            logs = [dialogue.split('+++$+++')[-1]]
        for i in range(len(logs)):
            s = logs[i]
            s = s.replace('</s> ', '')\
            .replace('<first_speaker> ', '')\
            .replace('<second_speaker> ', '')\
            .replace('<third_speaker>', '')\
            .replace('<at> ', '')\
            .replace('</d> ', '')\
            .replace(' </s>', '')\
            .replace(' </d>', '')\
            .replace('__eou__ ', '')\
            .replace(' __eou__', '')\
            .replace(';', '')\
            .replace('__eot__ ', '').strip()
            s = re.sub('<speaker_[0-9]+> ', '', s)
            s = unicode(s, errors='ignore')
            logs[i] = s
        str_list = list(filter(None, logs)) # fastest
        tweets.extend(str_list)
    return tweets

def transform_csv(fname):
    turns = []
    with open(fname, 'r') as fp:
        lines = csv.reader(fp)
        for idx, line in enumerate(lines):
            # Skip header line
            if idx == 0:
                continue

            assert len(line) == 2
            context = line[0]
            response = line[1]

            turns.append(context)
            turns.append(response)
    return turns

def add_capped_priority(queue, items, cap=300):
    for item in items:
        heapq.heappush(queue, item)
    while len(queue) > cap:
        heapq.heappop(queue)

def dump_pr_q(queue, outfile):
    with open(outfile, 'wt') as f:
        for x in queue:
            f.write(x + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, help="data set name to analyse", default=None,
                        choices=['twitter', 'reddit', 'ubuntu', 'movie',
                                 'hredstoch', 'hredbeam', 'vhredstoch', 'vhredbeam'])
    parser.add_argument('--input', type=str, help="txt or csv file to analyse", default=None)
    args = parser.parse_args()

    if args.data_name == 'twitter':
        data_paths = [
            '/home/ml/nangel3/research/data/twitter/train.txt',
            '/home/ml/nangel3/research/data/twitter/valid.txt',
            '/home/ml/nangel3/research/data/twitter/test.txt'
        ]
        utterance_delimiter='</s>'
    elif args.data_name == 'reddit':
        data_paths = [
            '/home/ml/nangel3/research/data/reddit/allnews/allnews_train.txt',
            '/home/ml/nangel3/research/data/reddit/allnews/allnews_val.txt',
            '/home/ml/nangel3/research/data/reddit/allnews/allnews_test.txt'
        ]
        utterance_delimiter='</s>'
    elif args.data_name == 'ubuntu':
        data_paths = [
            '/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_training_text.txt',
            '/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_valid_text.txt',
            '/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_test_text.txt'
        ]
        utterance_delimiter='__eou__'
    elif args.data_name == 'movie':
        # extracted raw movie dialogs to
        data_paths = ['/home/ml/nangel3/research/data/cornell_movie-dialogs_corpus/movie_lines.txt']
        utterance_delimiter=None
    elif args.data_name == 'hredbeam':
        data_paths = ['/home/ml/nangel3/research/data/twitter/ModelResponses/HRED/HRED_20KVocab_BeamSearch_5_GeneratedTrainResponses_TopResponse.txt']
        utterance_delimiter=None
    elif args.data_name == 'hredstoch':
        data_paths = ['/home/ml/nangel3/research/data/twitter/ModelResponses/HRED/HRED_Stochastic_GeneratedTrainResponses.txt']
        utterance_delimiter=None
    elif args.data_name == 'vhredbeam':
        data_paths = ['/home/ml/nangel3/research/data/twitter/ModelResponses/VHRED/VHRED_5000BPE_BeamSearch_5_GeneratedTrainResponses_TopResponse.txt']
        utterance_delimiter=None
    elif args.data_name == 'vhredstoch':
        data_paths = ['/home/ml/nangel3/research/data/twitter/ModelResponses/VHRED/VHRED_5000BPE_Stochastic_GeneratedTrainResponses.txt']
        utterance_delimiter=None
    else:
        data_paths = []
        utterance_delimiter = None
        print("WARNING: no dataset recognized.")

    if args.input:
        data_paths.append(args.input)

    assert len(data_paths) > 0


    print "Loading data to classify..."
    data = []
    for x in data_paths:
        if x.endswith('.txt'):
            data.extend(transform_twitter(x, utterance_delimiter, args.data_name == 'movie'))
        else:
            data.extend(transform_csv(x))
        print("Loaded %d statements" % len(data))
        print("Examples : %s ; %s ; %s" % tuple(data[:3]))
    import gc; gc.collect()

    print "Loading trained classifier... "
    model = joblib.load('final_model.pkl')

    print "Loading other information..."
    tf_vectorizer = joblib.load('final_tfidf.pkl')
    idf_vector = joblib.load('final_idf.pkl')
    pos_vectorizer = joblib.load('final_pos.pkl')
    #Load ngram dict
    #Load pos dictionary
    #Load function to transform data
    q = []
    q_off = []

    print "Transforming inputs..."
    batch_size=10000
    vals = []
    for start_idx in range(0, len(data), batch_size):
        stop_idx = min(len(data), start_idx+batch_size)
        X = transform_inputs(data[start_idx:stop_idx], tf_vectorizer, idf_vector, pos_vectorizer)

        print "Running classification model..."
        y = predictions(X, model)
        hate_speech_idxs = np.where(y==0)[0]
        offensive_speech_idxs = np.where(y==1)[0]
        q.extend(np.array(data[start_idx:stop_idx])[hate_speech_idxs])
        q_off.extend(np.array(data[start_idx:stop_idx])[offensive_speech_idxs])
        vals.extend(y)

    if args.data_name:
        dump_pr_q(q, args.data_name + "debughate.csv")
        dump_pr_q(q_off, args.data_name + "debugoffensive.csv")
    else:
        dump_pr_q(q, args.input.split('/')[-1] + "debughate.csv")
        dump_pr_q(q_off, args.input.split('/')[-1] + "debugoffensive.csv")
    print("%d examples (%f percent) hate speech and %d examples (%f percent) offensive language" % (len(q), float(len(q))/float(len(data))* 100., len(q_off), float(len(q_off))/float(len(data))*100.0))

