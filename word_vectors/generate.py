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
args = parser.parse_args()

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    print 'Loading model {}'.format(args.checkpoint)
    model = torch.load(f)


corpus = data.Corpus(args.data, load=True, prefix='google')
ntokens = len(corpus.dictionary)


def generate_sentence(starting_token):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
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
        response.append(word)
    response = ' '.join(response)
    return response


samples = None

if args.start_samples:
    samples = pd.read_csv(args.start_samples)
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
