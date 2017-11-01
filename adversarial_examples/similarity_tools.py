# Cosine similarity between two sentences using Spacy
import spacy
from siamese_lstm.lstm import lstm


class SimilarityTools:
    def __init__(self):
        self.nlp = spacy.load('en')
        self.siamese = lstm("siamese_lstm/bestsem.p",
                            load=True, training=False)

    def cosine_similarity(self, sent1, sent2):
        sent1 = self.nlp(unicode(sent1))
        sent2 = self.nlp(unicode(sent2))
        return sent1.similarity(sent2)

    def lstm_similarity(self, sent1, sent2):
        return self.siamese.predict_similarity(sent1, sent2) * 4.0 + 1.0

    def run_similarity(self, pairs):
        """Given sentence pairs, run similariy on all metrics
        """
        sims = []
        for pair in pairs:
            cosine_sim = self.cosine_similarity(pair[0], pair[1])
            lstm_sim = self.lstm_similarity(pair[0], pair[1])
            sims.append({
                'sent1': pair[0], 'sent2': pair[1],
                'cosine_sim': cosine_sim, 'lstm_sim': lstm_sim
            })
        return sims

if __name__ == '__main__':
    sents = [('He is my best friend','I consider him as my best buddy')]
    sim_tools = SimilarityTools()
    sims = sim_tools.run_similarity(sents)
    print sims
