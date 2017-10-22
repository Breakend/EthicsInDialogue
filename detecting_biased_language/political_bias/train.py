import string
from argparse import ArgumentParser
import cPickle as pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from collections import defaultdict
import sys
import pandas as pd
from random import shuffle
from time import strftime
from time import time
import logging
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import svm
from nltk.corpus import stopwords

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document.decode('utf-8').strip()):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)] 

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class BiasClassifier(object):

    def __init__(self, model=None, train=True, train_data=None,
                 dump=False, debug=False, crossval=False):
        """Intialize classifier, either from pre-trained model or from scratch"""
        self.debug = debug
        if crossval:
            self.pipeline_1 = self.train(train_data, crossval=crossval)
        else:
            if model:
                try:
                    self.pipeline_1 = self.load_model(model)
                    self.model_name = model
                except Exception as e_load:
                    logging.critical(str(e_load))
                    self.classifier = None
            else:
                self.pipeline_1 = self.train(train_data)

                if dump:
                    self.dump_model()

    def load_model(self, model_file=None):
        """ Load model from pre-trained pickle"""
        if self.debug:
            logging.info("Loading model %s" % model_file)
        try:
            with open(model_file, "rb") as pkl:
                pipeline = pickle.load(pkl)
        except (IOError, pickle.UnpicklingError) as e:
            logging.critical(str(e))
            raise e
        return pipeline

    def dump_model(self, model_file="model_%s.pkl" % strftime("%Y%m%d_%H%M")):
        """ Pickle trained model """
        if self.debug:
            logging.info("Dumping model to %s" % model_file)
        with open(model_file, "wb") as f_pkl:
            try:
                pickle.dump(self.pipeline_1, f_pkl, pickle.HIGHEST_PROTOCOL)
                self.model_name = model_file
            except pickle.PicklingError as e_pkl:
                print str(e_pkl) + ": continuing without dumping."

    def create_pipeline(self):
        pipeline = Pipeline([
             ('ngrams_text', Pipeline([
                #  ('selector', ItemSelector(key='Text')),
                ('preprocessor', NLTKPreprocessor()),
                 ('vect', TfidfVectorizer(ngram_range=(1,3), preprocessor=None, lowercase=False, tokenizer=lambda x: x)),
             ])),
            #('logreg', LogisticRegression(penalty="l2", C=1.5, dual = True,  class_weight=None)),
            ('logreg', svm.LinearSVC(C=1.5, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0))
            ])
        return pipeline

    def train(self, train_path, crossval=False):
        """ Train classifier on features from headline and article text """
        if self.debug:
            tick = time()
            logging.info("Training new model with %s" % (train_path,))
            logging.info("Loading/shuffling training data...")

        train_data_1 = pd.read_csv(train_path)
        #shuffle(train_data_1)
        train_data_1 = train_data_1.sample(frac=1).reset_index(drop=True)

        pipeline_1 = self.create_pipeline()

        if crossval:
            scores = cross_validation.cross_val_score(pipeline_1, train_data_1['Text'], train_data_1['BiasScore'], cv=5)
            print("5 fold scores")
            print(scores)
        else:
            pipeline_1.fit(train_data_1['Text'], train_data_1['BiasScore'])

        return pipeline_1


    def classify(self, inputs):
        """ Classifies inputs """
        responses = []
        prediction = self.pipeline_1.predict(inputs)
        return responses

def main():
    logging.basicConfig(level=logging.INFO)

    argparser = ArgumentParser(description=__doc__)
    argparser.add_argument("-t", "--trainset", action="store",
                           default=None,
                           help=("Path to training data "
                                 "[default: %(default)s]"))
    argparser.add_argument("-m", "--model", action="store",
                           help="Path to model")
    argparser.add_argument("-d", "--dump", action="store_true",
                           help="Pickle trained model? [default: False]")
    argparser.add_argument("-v", "--verbose", action="store_true",
                           default=False,
                           help="Verbose [default: quiet]")
    argparser.add_argument("-c", "--classify", action="store",
                           default=None,
                           help=("Path to data to classify "
                                 "[default: %(default)s]"))
    argparser.add_argument("-s", "--save", action="store",
                           default='output.csv',
                           help=("Path to output file"
                                 "[default = output.csv]"))
    argparser.add_argument("--crossval", action="store_true")
    args = argparser.parse_args()


    clf = BiasClassifier(train_data=args.trainset,
                                    model=args.model,
                                    dump=args.dump,
                                    debug=args.verbose,
                                    crossval=args.crossval)

    if args.classify:
        raise NotImplementedError

if __name__ == "__main__":
    main()
