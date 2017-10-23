###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--token', type=str, default='she',
                    help='start token to generate sentences')
parser.add_argument('--start_samples', type=str,
                    help='csv file containing the start tokens')
parser.add_argument('--header_column', type=str,
                    help='csv file header')
parser.add_argument('--stochastic', action='store_true',
                    help='Run the stochastic experiment')
parser.add_argument('--stoch_times', type=int, default=1000,
                    help='Run the stochastic experiment n times')
args = parser.parse_args()

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    print 'Loading model {}'.format(args.checkpoint)
    model = torch.load(f)


corpus = data.Corpus(args.data, load=True, prefix='google')
ntokens = len(corpus.dictionary)

# gender centric words
male_words = ["he", "him", "his", "himself", "husband", "man", "men", "boy"]
female_words = ["she", "her", "herself", "wife", "woman", "women", "girl"]


def set_seed():
    """ Set the random seed manually for reproducibility.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


def generate_sentence(starting_token, fix_seed=True):
    if fix_seed:
        set_seed()
    cmodel = copy.deepcopy(model)
    if args.cuda:
        cmodel.cuda()
    else:
        cmodel.cpu()
    cmodel.eval()
    words = starting_token.lower().strip().split(' ')
    start_word = [[corpus.dictionary.word2idx[words[0]]]]
    input = Variable(torch.LongTensor(start_word), volatile=True)
    # input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    if args.cuda:
        input.data = input.data.cuda()
    response = []
    response.append(words[0])
    hidden = cmodel.init_hidden(1)
    for i in range(args.words):
        output, hidden = cmodel(input, hidden)
        if (i + 1) < len(words):
            word_idx = corpus.dictionary.word2idx[words[i + 1]]
        else:
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            #word_weights = word_weights.unsqueeze(0)
            # print word_weights.size()
            word_idx = torch.multinomial(word_weights, 1)[0]
            #maxes, indices = torch.max(word_weights, -1)
            # word_idx = indices[0]  # .numpy()[0]
            # print maxes
            # print word_idx

        input.data.fill_(word_idx)
        # print input
        word = corpus.dictionary.idx2word[word_idx]
        # break on end of sentence
        if word == '<eos>':
            break
        response.append(word)
    response = ' '.join(response)
    return response


samples = None

if args.start_samples:
    samples = pd.read_csv(args.start_samples)
    if args.stochastic:
        samples['male_percentage'] = 0.0
        samples['female_percentage'] = 0.0
        samples['unbiased_percentage'] = 0.0
        pb = tqdm(total=len(samples))
        # run the stochastic experiment
        for i, row in samples.iterrows():
            # set the seed first and then dont
            set_seed()
            male_count = 0
            female_count = 0
            pbs = tqdm(total=args.stoch_times)
            for step in range(args.stoch_times):
                response = generate_sentence(
                    row['token'], fix_seed=False).split(' ')
                for word in male_words:
                    if word in response:
                        male_count += 1
                        break
                for word in female_words:
                    if word in response:
                        female_count += 1
                        break
                pbs.update(1)
            pbs.close()
            if male_count + female_count > args.stoch_times:
                extra = male_count + female_count - args.stoch_times
                male_count = male_count - extra
                female_count = female_count - extra
            unbiased = args.stoch_times - (male_count + female_count)
            male_perc = (1.0 * male_count) / args.stoch_times
            female_perc = (1.0 * female_count) / args.stoch_times
            unbiased_perc = (1.0 * unbiased) / args.stoch_times
            samples.set_value(i, 'male_percentage', male_perc)
            samples.set_value(i, 'female_percentage', female_perc)
            samples.set_value(i, 'unbiased_percentage', unbiased_perc)
            pb.update(1)
        pb.close()
        samples.to_csv(args.start_samples, encoding='utf-8', index=None)
    else:
        # run the sample generation experiment
        samples[args.header_column] = ''

        if len(samples) > 0:
            for i, row in samples.iterrows():
                token = row['token']
                try:
                    response = generate_sentence(token)
                    samples.set_value(i, args.header_column, response)
                except Exception as e:
                    print e
                print 'Generated {} / {} sentence'.format(i, len(samples))
            samples.to_csv(args.start_samples, encoding='utf-8', index=None)
else:
    token = args.token
    response = generate_sentence(token)
    print response
