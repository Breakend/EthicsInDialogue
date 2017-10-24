import numpy as np
import argparse


def get_bias_stats(data, feat):

    sentences = [str(data[i].split(',')[0]) for i in range(1, len(data)-2)]
    sentences = np.array(sentences)

    if feat == 'bias':
        idx = np.where(np.array(data[0].split(',')) == 'bias_score')[0][0]
    elif feat == 'subjectivity':
        idx = np.where(np.array(data[0].split(',')) == 'subjectivity_score')[0][0]
    elif feat == 'vador':
        idx = np.where(np.array(data[0].split(',')) == 'vader_composite_sentiment')[0][0]
    elif feat == 'unread':
        idx = np.where(np.array(data[0].split(',')) == 'flesch-kincaid_grade_level')[0][0]
    else:
        print "ERROR: unknown feature %s" % feat
        return

    vals = [float(data[i].split(',')[idx]) for i in range(1, len(data)-2)]
    vals = np.array(vals)

    max_idx = np.argmax(vals)
    min_idx = np.argmin(vals)

    return np.mean(vals), (np.max(vals), sentences[max_idx]), (np.min(vals), sentences[min_idx]), np.std(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', choices=['twitter', 'reddit', 'ubuntu', 'movie', 'hred_twitter_stoch', 'hred_twitter_beam5', 'vhred_twitter_stoch', 'vhred_twitter_beam5'])
    args = parser.parse_args()

    print "loading data..."
    with open('%s.csv' % args.data_name, 'r') as handle:
        data = handle.readlines()
    print "%d lines" % len(data)

    print "computing avg/max/min/std..."
    for feat in ['bias', 'subjectivity', 'vador', 'unread']:
        data_avg, (data_max, sentence_max), (data_min, sentence_min), data_std = get_bias_stats(data, feat)
        print "[%s] avg: %f" % (feat, data_avg)
        print "[%s] max: %f -- sentence: %s" % (feat, data_max, sentence_max)
        print "[%s] min: %f -- sentence: %s" % (feat, data_min, sentence_min)
        print "[%s] std: %f" % (feat, data_std)
        print ""


if __name__ == '__main__':
    main()


