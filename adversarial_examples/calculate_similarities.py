# Calculate similarity scores w.r.t base sentences and adversarial sentences
import similarity_tools as st
import glob
import re
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize

MOVIES_DIR = './data/movies/'
POLITICS_DIR = './data/politics/'
DATA_DIRS = {'movies': MOVIES_DIR, 'politics': POLITICS_DIR}
EDIT_TYPES = ['paraphrased', 'characteredits']
FILE_TYPES = {'paraphrased': 'paraphrased', 'characteredits': 'edits'}
RESPONSE_PREFIX = 'response_'

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


def get_base_files(base_dir):
    file_dir = {}
    with open(base_dir + 'base_sentences.txt', 'r') as fp:
        for line_indx, line in enumerate(fp):
            file_dir[line_indx] = {'base': line}
    return file_dir


def get_adversarial_files(base_dir, file_dir, edit_type):
    pfiles = glob.glob(base_dir + edit_type + '/' +
                       FILE_TYPES[edit_type] + '*.txt')
    rfiles = glob.glob(base_dir + edit_type + '/' +
                       RESPONSE_PREFIX + FILE_TYPES[edit_type] + '*.txt')
    sort_nicely(pfiles)
    sort_nicely(rfiles)
    for indx, pfile in enumerate(pfiles):
        plist = []
        rlist = []
        rfile = rfiles[indx]
        with open(pfile, 'r') as fp:
            for line in fp:
                plist.append(line)
        with open(rfile, 'r') as fp:
            for line in fp:
                rlist.append(line)
        file_dir[indx][edit_type] = {
            'base_list': plist, 'response_list': rlist}
    return file_dir


def tokenize_join(sent):
    return ' '.join(word_tokenize(sent))


def calc_sims(sim_tool, file_dir, edit_type):
    print 'Calculating similarity for {}, type {}'.format(file_dir, edit_type)
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


if __name__ == '__main__':
    sim_tool = st.SimilarityTools()
    for base_key, base_dir in DATA_DIRS.iteritems():
        for edit_type in EDIT_TYPES:
            file_dir = get_base_files(base_dir)
            file_dir = get_adversarial_files(MOVIES_DIR, file_dir, edit_type)
            file_dir = calc_sims(sim_tool, file_dir, edit_type)
            json.dump(file_dir, open(
                base_key + '_' + edit_type + '.json', 'w'))
