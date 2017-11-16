# Language Modelling Experiment with Word Vectors

This codebase is forked from [Pytorch's Language Modelling](https://github.com/pytorch/examples/tree/master/word_language_model) example.
This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

## Experiments

### Training

Using pretrained [word2vec](https://code.google.com/archive/p/word2vec/) and [debiased](https://github.com/tolga-b/debiaswe) word embeddings.
Run the code using the following args:

```bash
python main.py --data ./data/google --cuda --epochs 10 --emsize=300 --nhid=650 --dropout 0.5 --type=debiased --save model_deb_fn.pt
```

Where, `--type` can be `debiased`, `word2vec`, `glove` and `concept`, and `--data` can be `./data/penn` for Penn Tree Bank dataset, or `./data/google` for Google 1Billion words dataset.

The above hyperparams produce 95 perplexity on Google 1B dataset.


### Generate

To generate examples:

```bash
python generate.py --checkpoint model_deb_fn.pt --words=100 --cuda --token "boss expects me to" && cat generated.txt
```

To generate both word2vec and debiased examples together:

```bash
./test_models_single.sh
```

To batch generate word2vec and debiased using input csv file `input_tabs.csv`:

```bash
./test_models.sh
```

## Acknowledgements

Forked from [Pytorch Language modelling example](https://github.com/pytorch/examples/tree/master/word_language_model).


