read inp
echo "--------word2vec-----------"
python generate.py --checkpoint model_word2vec_goog.pt --words=20 --cuda --data ./data/google --start_samples input_tabs.csv --header_column word2vec
echo "--------word2vec new ---------"
python generate.py --checkpoint model_word2vec_goog_lr5_bk.pt --words=20 --cuda --data ./data/google --start_samples input_tabs.csv --header_column word2vec_new
echo "--------debiased-----------"
python generate.py --checkpoint model_deb_lin.pt --words=20 --cuda --data ./data/google --start_samples input_tabs.csv --header_column debiased
echo "--------debiased new ------"
python generate.py --checkpoint model_deb_lin_2_bk.pt --words=20 --cuda --data ./data/google --start_samples input_tabs.csv --header_column debiased_new

