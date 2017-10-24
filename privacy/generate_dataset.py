import re
import random

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
key_type = 'nl'
num_keypairs = 10
key_pairs = [] 
if key_type == 'uuid':
    import uuid
    for i in range(num_keypairs):
        key_pairs.append((str(uuid.uuid1()), str(uuid.uuid1())))
elif key_type == 'nl':
    # natural language
    assert num_keypairs <= 10
    key_list = [ "pete", "bob", "nic", "alice", "george", "rose", "jane", "jack", "john", "jill", "xi", "jin", "vincent", "eve"]
    pass_list = [ "metal", "fuse", "telescope", "gown", "sensation", "contract", "certificate", "variation", "contributor", "modeling", "prison", "rain", "underlay"] 
    for i in range(num_keypairs):
        key_pairs.append(("my name is %s ." % key_list[i], "my password is %s ." % pass_list[i])) 
elif key_type == 'subsampled':
    key_list = [ "pete", "bob", "nic", "alice", "george", "rose", "jane", "jack", "john", "jill", "xi", "jin", "vincent", "eve"]
    for i in range(num_keypairs):
        key_pairs.append(("my name is %s ." % key_list[i], "my password is %s ." % random.choice(vocab_target)))
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

