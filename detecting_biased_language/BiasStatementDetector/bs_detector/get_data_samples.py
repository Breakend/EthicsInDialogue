import numpy as np
import argparse
from texttable import Texttable
import csv


DATA_NAME_TO_FORMAT = {
    'twitter': 2,  # skip last two lines of csv file: (total & average)
    'ubuntu': 2,   # skip last two lines of csv file: (total & average)
    'movie': 2,    # skip last two lines of csv file: (total & average)
    'reddit': 2,   # skip last two lines of csv file: (total & average)
    'hred_twitter_stoch': 5,  # skip last five lines of csv file: (total & average & max & min & std)
    'hred_twitter_beam5': 5,  # skip last five lines of csv file: (total & average & max & min & std)
    'vhred_twitter_stoch': 5,  # skip last five lines of csv file: (total & average & max & min & std)
    'vhred_twitter_beam5': 5,  # skip last five lines of csv file: (total & average & max & min & std)
}

def get_samples(data_name, data, feats, k):

    start = 1
    if data_name == 'reddit':
        start = 3300000  # reddit.csv file messed up the first 3.3 million lines, skip those... -_-

    # list of sentences
    sentences = [str(data[i].split(',')[0]) for i in range(start, len(data)-DATA_NAME_TO_FORMAT[data_name])]

    # column index of each feature we care about
    feat_idx = []
    if 'bias' in feats:
        idx = np.where(np.array(data[0].split(',')) == 'bias_score')[0][0]
        feat_idx.append(idx)
    if 'subjectivity' in feats:
        idx = np.where(np.array(data[0].split(',')) == 'subjectivity_score')[0][0]
        feat_idx.append(idx)
    if 'vader' in feats:
        idx = np.where(np.array(data[0].split(',')) == 'vader_composite_sentiment')[0][0]
        feat_idx.append(idx)
    if 'flesch-kincaid' in feats:
        idx = np.where(np.array(data[0].split(',')) == 'flesch-kincaid_grade_level')[0][0]
        feat_idx.append(idx)
    if len(feat_idx) == 0:
        print "ERROR: unknown features %s" % feats
        return

    feat_vals = []  # list of feature values for each sentence

    for i in range(start, len(data)-DATA_NAME_TO_FORMAT[data_name]):
        try:
            feat_vals.append( [float(data[i].split(',')[idx]) for idx in feat_idx] )
        except ValueError as e:
            # remove that sentence
            del sentences[i]
            # skip examples: when data[i].split()[idx] is a string and not a float
            continue

    sentences = np.array(sentences)
    feat_vals = np.array(feat_vals)
    print "collected features: %s" % (feat_vals.shape,)
    assert len(feat_vals) == len(sentences)

    sampled_idx = np.random.choice(range(len(sentences)), size=k, replace=False)
    sampled_sents = sentences[sampled_idx]
    sampled_feats = feat_vals[sampled_idx]

    return sampled_sents, sampled_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', choices=['twitter', 'reddit', 'ubuntu', 'movie', 'hred_twitter_stoch', 'hred_twitter_beam5', 'vhred_twitter_stoch', 'vhred_twitter_beam5'])
    parser.add_argument('--k', type=int, default=1000, help="number of sentences to sample from dataset")
    args = parser.parse_args()

    print "loading data..."
    with open('%s/%s.csv' % (args.data_name, args.data_name), 'r') as handle:
        data = handle.readlines()
    print "%d lines" % len(data)

    features = ['bias', 'subjectivity', 'vader', 'flesch-kincaid']
    header = ['sentences'] + features
    print "sampling %d sentences and computing %s..." % (args.k, features)
    sampled_sents, sampled_feats = get_samples(args.data_name, data, features, args.k)
    sampled = np.concatenate((sampled_sents[:,np.newaxis], sampled_feats), axis=1)
    with open('%s/%s_%dsampled.csv' % (args.data_name, args.data_name, args.k), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for row in sampled:
            writer.writerow(row)

    with open('%s/%s_%dsampled.md' % (args.data_name, args.data_name, args.k), 'w') as fp:
        page_header = '''
---
layout: project_page
title: Sampled sentences with their (bias, subjectivity, vader, and unreadable) scores
---

        '''
        headers = '|'.join(header) + '\n'
        sep = '|'.join(['-' * len(head) for head in header]) + '\n'
        tds = '\n'.join(['|'.join([str(item) for item in row]) for row in sampled])
        fp.write(page_header + '\n' + headers + sep + tds + '\n')

    t = Texttable()
    rows = [header]
    rows.extend(sampled)
    t.add_rows(rows)
    print t.draw()


if __name__ == '__main__':
    main()


