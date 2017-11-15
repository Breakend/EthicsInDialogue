import numpy as np
import argparse


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

def get_bias_stats(data_name, data, feats):

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
    if 'unread' in feats:
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

    max_idx = np.argmax(feat_vals, axis=0)  # array of max indices for each feature
    min_idx = np.argmin(feat_vals, axis=0)  # array of min indices for each feature

    return np.mean(feat_vals, axis=0), (np.max(feat_vals, axis=0), sentences[max_idx]), (np.min(feat_vals, axis=0), sentences[min_idx]), np.std(feat_vals, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', choices=['twitter', 'reddit', 'ubuntu', 'movie', 'hred_twitter_stoch', 'hred_twitter_beam5', 'vhred_twitter_stoch', 'vhred_twitter_beam5'])
    args = parser.parse_args()

    print "loading data..."
    with open('%s/%s.csv' % (args.data_name, args.data_name), 'r') as handle:
        data = handle.readlines()
    print "%d lines" % len(data)

    features = ['bias', 'subjectivity', 'vader', 'unread']
    print "computing avg/max/min/std on %s..." % features
    data_avg, (data_max, sentence_max), (data_min, sentence_min), data_std = get_bias_stats(args.data_name, data, features)
    for idx, feat in enumerate(features):
        print "[%s] avg: %f" % (feat, data_avg[idx])
        print "[%s] max: %f -- sentence: %s" % (feat, data_max[idx], sentence_max[idx])
        print "[%s] min: %f -- sentence: %s" % (feat, data_min[idx], sentence_min[idx])
        print "[%s] std: %f" % (feat, data_std[idx])
        print ""


if __name__ == '__main__':
    main()


