read inp
echo "--------word2vec-----------"
python generate.py --checkpoint model_word2vec_goog.pt --words=20 --cuda --data ./data/google --token "$inp"
echo "--------debiased-----------"
python generate.py --checkpoint model_deb_lin.pt --words=20 --cuda --data ./data/google --token "$inp"

