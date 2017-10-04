# coding: utf-8
import pandas
data = pandas.read_csv('5gram-edits-test.tsv', sep='\t*', header=None, error_bad_lines=False)
get_ipython().magic('ls ')
data
data[0]
data
data[3]
data[4]
data[2]
data[3][True]
data[data[3] == True]
data[data[3] == True][0]
data[data[3] == True][-1]
data[data[3] == True]
data[data[3] == True][9]
data[data[3] == True][7]
data[data[3] == True][8]
data[data[3] == True][8][0]
data[data[3] == True][8]
data[data[3] == True][8][340]
data[data[3] == True][9][340]
data[data[3] == True][9][340]
data.to_csv('converted_dev.csv')
data.to_csv('converted_test.csv')
data = pandas.read_csv('5gram-edits-dev.tsv', sep='\t*', header=None, error_bad_lines=False)
data = data[data[3] == True]
data.to_csv('converted_dev.csv')
data = pandas.read_csv('5gram-edits-test.tsv', sep='\t*', header=None, error_bad_lines=False)
data = data[data[3] == True]
data.to_csv('converted_test.csv')
data = pandas.read_csv('5gram-edits-train.tsv', sep='\t*', header=None, error_bad_lines=False)
data = data[data[3] == True]
data.to_csv('converted_train.csv')
data.shape
data = pandas.read_csv('5gram-edits-train.tsv', sep='\t', header=None, error_bad_lines=False)
data = data[data[3] == True]
data.shape
get_ipython().magic('save current_session ~0/')
