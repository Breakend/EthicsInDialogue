# Adversarial Data

The goal of this experiment is to demonstrate that generative models are not robust to small changes in input data which can lead to divergent distributions of output data.

To do this we use 3 metrics: edit distance, sentence similarity [1], sentence similarity [2].


We have 20 base sentences which we input a VHRED retrieval model for Movies and for Politics [4].

For character-level perturbations, we generate 1000 input adversarial sentences per base sentence where a character is either removed, added, or mutated at random.

For word-level adversarial examples, we hand-write 6 paraphrased sentences per base sentence.

For each, we measure the edit distance and the similarities of our input metrics. Then we assess the distance/similarity from the base VHRED response.


[1] TODO: nltk or sklearn edit distance here
[2] https://raw.githubusercontent.com/sujitpal/nltk-examples/master/src/semantic/short_sentence_similarity.py
[3] https://github.com/aditya1503/Siamese-LSTM
[4] MILABOT TODO.
