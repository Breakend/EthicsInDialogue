# Calculate similarity scores w.r.t base sentences and adversarial sentences
import sys
import os
import similarity_tools as st
import glob
import re
import json
import csv
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

MOVIES_DIR = './data/movies/'
POLITICS_DIR = './data/politics/'
DATA_DIRS = {'movies': MOVIES_DIR, 'politics': POLITICS_DIR}
EDIT_TYPES = ['paraphrased', 'characteredits']
FILE_TYPES = {'paraphrased': 'paraphrased', 'characteredits': 'edits'}
RESPONSE_PREFIX = 'response_'
CNN_SIM_DATA_PATH = './cnn_similarity/data/'

# string list sort (https://nedbatchelder.com/blog/200712/human_sorting.html)


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def get_base_files(base_dir, mode=''):
    """ Mode : none or response
    """
    file_dir = {}
    filename = base_dir + 'base_sentences.txt'
    if mode == 'response':
        filename = base_dir + 'response_base_sentences.txt'
    with open(filename, 'r') as fp:
        for line_indx, line in enumerate(fp):
            file_dir[line_indx] = {'base': line}
    return file_dir


def get_adversarial_files(base_dir, file_dir, edit_type, mode=''):
    if 'politics' in base_dir:
        resp_prefix = 'politics_' + RESPONSE_PREFIX
    else:
        resp_prefix = RESPONSE_PREFIX
    if mode != 'response':
        pfiles = glob.glob(base_dir + edit_type + '/' +
                           FILE_TYPES[edit_type] + '*.txt')
    else:
        pfiles = glob.glob(base_dir + edit_type + '/' +
                           resp_prefix + FILE_TYPES[edit_type] + '*.txt')
    sort_nicely(pfiles)
    for indx, pfile in enumerate(pfiles):
        plist = []
        with open(pfile, 'r') as fp:
            for line in fp:
                plist.append(line)
        file_dir[indx][edit_type] = {
            'list': plist}
    return file_dir


def tokenize_join(sent):
    return ' '.join(word_tokenize(sent))


def calc_sims(sim_tool, df):
    """
    print 'Calculating similarity for {}'.format(edit_type)
    pb = tqdm(total=len(file_dir))
    for indx in file_dir:
        sim_inp_par = [(tokenize_join(file_dir[indx]['base']),
                        tokenize_join(line))
                       for line in file_dir[indx][edit_type]['base_list']]
        sim_par_res = sim_tool.run_similarity(sim_inp_par)
        file_dir[indx][edit_type]['cosine_sims'] = {
            'base_list': [], 'response_list': []}
        file_dir[indx][edit_type]['lstm_sims'] = {
            'base_list': [], 'response_list': []}
        for res in sim_par_res:
            file_dir[indx][edit_type]['cosine_sims']['base_list'].append(
                res['cosine_sim'])
            file_dir[indx][edit_type]['lstm_sims']['base_list'].append(
                res['lstm_sim'][0])
        sim_inp_par = [(tokenize_join(file_dir[indx]['base']),
                        tokenize_join(line))
                       for line in file_dir[indx][edit_type]['response_list']]
        sim_par_res = sim_tool.run_similarity(sim_inp_par)
        for res in sim_par_res:
            file_dir[indx][edit_type]['cosine_sims']['response_list'].append(
                res['cosine_sim'])
            file_dir[indx][edit_type]['lstm_sims']['response_list'].append(
                res['lstm_sim'][0])

        pb.update(1)
    pb.close()
    return file_dir
    """
    pairs = []
    for i, row in df.iterrows():
        pairs.append((row[0], row[1]))
    sim_par_res = sim_tool.run_similarity(pairs)
    df['cosine'] = 0.0
    df['lstm'] = 0.0
    for i, res in enumerate(sim_par_res):
        df.set_value(i, 'cosine', res['cosine_sim'])
        df.set_value(i, 'lstm', res['lstm_sim'][0])
    return df


def prepare_files_cnn_similarity(base_dir, file_dir, edit_type, vocab):
    sim_inp_par = []
    for indx, vals in file_dir.iteritems():
        sim_inp_par.extend([(tokenize_join(file_dir[indx]['base']),
                             tokenize_join(line), base_dir, edit_type)
                            for line in file_dir[indx][edit_type]['list']])
        for pair in sim_inp_par:
            words = pair[0].split(' ') + pair[1].split(' ')
            vocab.update(words)
    return sim_inp_par, vocab


if __name__ == '__main__':
    sim_tool = st.SimilarityTools()
    sim_pairs = []
    vocab = Counter()
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'context'
    print "Selecting {}".format(mode)
    """
    for base_key, base_dir in DATA_DIRS.iteritems():
        for edit_type in EDIT_TYPES:
            file_dir = get_base_files(base_dir, mode)
            file_dir = get_adversarial_files(
                base_dir, file_dir, edit_type, mode)
            sm, vocab = prepare_files_cnn_similarity(
                base_dir, file_dir, edit_type, vocab)
            sim_pairs.extend(sm)
            # file_dir = calc_sims(sim_tool, file_dir, edit_type)
            #  json.dump(file_dir, open(
            #     base_key + '_' + edit_type + '.json', 'w'))
    path = CNN_SIM_DATA_PATH + 'ethics/'
    if not os.path.exists(path):
        os.makedirs(path)
    ewriter = csv.writer(open('pairs_{}.csv'.format(mode), 'w'), delimiter=',')
    ewriter.writerow(['sent1', 'sent2', 'base_dir', 'edit_type'])
    with open(path + '/a.toks', 'w') as fpa:
        with open(path + '/b.toks', 'w') as fpb:
            with open(path + '/id.txt', 'w') as fpid:
                for pair_id, pair in enumerate(sim_pairs):
                    fpid.write(str(pair_id) + '\n')
                    fpa.write(pair[0] + '\n')
                    fpb.write(pair[1] + '\n')
                    ewriter.writerow(pair)
    vocab = sorted(vocab.keys())
    with open(path + '/vocab_{}.txt'.format(mode), 'w') as fp:
        for word in vocab:
            fp.write(word + '\n')
    print "Calculating sims"
    """
    df = pd.read_csv('pairs_{}.csv'.format(mode))
    df = calc_sims(sim_tool, df)
    df.to_csv('pairs_{}_metrics.csv'.format(mode), index=None)
