import cPickle
import numpy as np

if __name__ == '__main__':
    [lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))

    # how to access sentence text
    print 'Liberal examples (out of ', len(lib), ' sentences): '
    for tree in lib[0:5]:
        print tree.get_words()

    print '\nConservative examples (out of ', len(con), ' sentences): '
    for tree in con[0:5]:
        print tree.get_words()

    print '\nNeutral examples (out of ', len(neutral), ' sentences): '
    for tree in neutral[0:5]:
        print tree.get_words()

    # how to access phrase labels for a particular tree
    ex_tree = lib[0]

    print '\nPhrase labels for one tree: '

    # see treeUtil.py for the tree class definition
    for node in ex_tree:

        # remember, only certain nodes have labels (see paper for details)
        if hasattr(node, 'label'):
            print node.label, ': ', node.get_words()

    datapoints = []
    datapoints.extend(lib)
    datapoints.extend(con)
    datapoints.extend(neutral)
    label_to_number = { "Neutral" : 0, "Conservative" : 1, "Liberal" : -1}
    newsaudito_to_number = { 1 : -1, 2 : 1, 0:0}
    import csv
    with open('lib.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar = '"')
        writer.writerow(('Text','BiasScore'))
        for x in datapoints:
            average_bias = np.round(np.mean([label_to_number[node.label] for node in x if hasattr(node, 'label')]))
            writer.writerow((x.get_words(), average_bias))

        import pandas as pd
        newsdata = pd.read_csv('reworked.csv')
        for index, row in newsdata.iterrows():
            writer.writerow((row['Text'], newsaudito_to_number[row['PoliticalLeaning']]))
                
