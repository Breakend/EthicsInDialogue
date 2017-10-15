import os
import torch
import torchwordemb


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def get_word_embeddings(self, embedding_file, save_name='glove_embeddings.mod', embedding_dim=300):
        """ Get word embeddings. Assumes a .txt file containing word and its vectors
        """
        print "Loading word embeddings from {}".format(embedding_file)
        assert os.path.exists(embedding_file)
        embeddings = torch.Tensor(len(self.dictionary), embedding_dim)
        with open(embedding_file, 'r') as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                if len(parsed) == 2:
                    # first line, so skip
                    continue
                assert(len(parsed) == embedding_dim + 1)
                # w = normalize_text(parsed[0])
                w = parsed[0]
                if w in self.dictionary.word2idx:
                    vec = [float(i) for i in parsed[1:]]
                    vec = torch.Tensor(vec)
                    embeddings[self.dictionary.word2idx[w]].copy_(vec)
        torch.save(embeddings, open(save_name, 'wb'))
        return embeddings

    def get_word_embeddings_bin(self, embedding_file, save_name='debiased_embeddings.mod', embedding_dim=300):
        """ Get word embeddings, where it assumes input as a bin file
        """
        print "Loading word embeddings from {}".format(embedding_file)
        assert os.path.exists(embedding_file)
        embeddings = torch.Tensor(len(self.dictionary), embedding_dim)
        vocab, vecs = torchwordemb.load_word2vec_bin(embedding_file)
        ct = 0
        for word in self.dictionary.word2idx:
            if word in vocab:
                v = vecs[vocab[word]]
                embeddings[self.dictionary.word2idx[word]].copy_(v)
                ct += 1
        print 'Copied {}/{} words'.format(ct, len(self.dictionary.word2idx))
        torch.save(embeddings, open(save_name, 'wb'))
        return embeddings

    def load_embeddings(self, embedding_file):
        return torch.load(open(embedding_file, 'rb'))
