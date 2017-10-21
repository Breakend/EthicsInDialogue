import numpy as np
import argparse


def get_bias_stats(data):

    sentences = [str(data[i].split(',')[0]) for i in range(1, len(data)-2)]
    sentences = np.array(sentences)

    bias = [float(data[i].split(',')[1]) for i in range(1, len(data)-2)]
    bias = np.array(bias)

    max_idx = np.argmax(bias)
    min_idx = np.argmin(bias)

    return (np.max(bias), sentences[max_idx]), (np.min(bias), sentences[min_idx]), np.std(bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', choices=['twitter', 'reddit', 'ubuntu', 'movie'])
    args = parser.parse_args()

    with open('%s.csv' % args.data_name, 'r') as handle:
        data = handle.readlines()

    (data_max, sentence_max), (data_min, sentence_min), data_std = get_bias_stats(data)
    print "max: %f -- sentence: %s" % (data_max, sentence_max)
    print "min: %f -- sentence: %s" % (data_min, sentence_min)
    print "std: %f" % data_std


if __name__ == '__main__':
    main()


