import re
import random


key_type = 'subsampled'

def get_list_from_file(file_name):
    with open(file_name, "r") as f1:
        l = f1.read().lower().split('\n')
    return l

def transform_twitter(twitter, utterance_delimiter='</s>', cornell=False):
    dialogues = get_list_from_file(twitter)
    tweets = []
    for dialogue in dialogues:
        if utterance_delimiter is None:
            logs = [dialogue]
        else:
            logs = dialogue.split(utterance_delimiter)
        if cornell:
            logs = [dialogue.split('+++$+++')[-1]]
        for i in range(len(logs)):
            s = logs[i]
            s = s.replace('<second_speaker> ', '')\
            .replace('<third_speaker>', '')\
            .replace('<at> ', '')\
            .replace('</d> ', '')\
            .replace(' </s>', '')\
            .replace('\t', ' ')\
            .replace(' </d>', '')\
            .replace(';', '')\
            .replace('__eot__ ', '').strip()
            s = re.sub('<speaker_[0-9]+> ', '', s)
            s = unicode(s, errors='ignore')
            logs[i] = s
        str_list = list(filter(None, logs)) # fastest
        tweets.extend(str_list)
    return tweets

ubuntu_training = transform_twitter("/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_training_text.txt", None)
subsampled = ubuntu_training[:10000]

vocab_source = []
vocab_target = []
for x in subsampled:
    utterances = x.split('__eou__')
    source = utterances[0]
    target = utterances[1]
    vocab_source.extend(source.split(" "))
    vocab_target.extend(target.split(" "))
num_keypairs = 10
key_pairs = [] 

with open('englishvocab.txt', 'r') as f:
    english_vocab = f.readlines()

if key_type == 'uuid':
    import uuid
    for i in range(num_keypairs):
        key_pairs.append((str(uuid.uuid1()), str(uuid.uuid1())))
elif key_type == 'nl':
    # natural language
    assert num_keypairs <= 10
    for i in range(num_keypairs):
        key_pairs.append((" ".join([x.strip() for x in random.sample(english_vocab, 5)]), " ".join([x.strip() for x in random.sample(english_vocab, 5)])))
elif key_type == 'subsampled':
    for i in range(num_keypairs):
        key_pairs.append((" ".join([x.strip() for x in random.sample(vocab_source, 5)]), " ".join([ x.strip() for x in random.sample(vocab_target, 5)]))) 
else:
    raise NotImplementedException     


suffix = key_type
subsampled.extend(["__eou__".join(x) for x in key_pairs])

with open('train%s.txt' % suffix, 'w') as f:
    vocab_source = []
    vocab_target = []
    for x in subsampled:
       utterances = x.split('__eou__') 
       source = utterances[0] 
       target = utterances[1]
       vocab_source.extend(source.split(" "))
       vocab_target.extend(target.split(" "))
       f.write("\t".join([source, target]) + "\n") 

with open('dev%s.txt' % suffix, 'w') as f:
    for x in key_pairs:
        f.write("\t".join(x) +"\n")

with open('vocab%s.source' % suffix, 'w') as f:
    for x in set(vocab_source):
        f.write(x + "\n")

with open('vocab%s.target' % suffix, 'w') as f:
    for x in set(vocab_target):
        f.write(x + "\n")

