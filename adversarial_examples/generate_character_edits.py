import argparse
import copy
import random
import string

parser = argparse.ArgumentParser()
parser.add_argument("base_sentences_file")
parser.add_argument("out_folder")
args = parser.parse_args()

num_edits_per_line = 1000



def character_insert(x):
    insert = random.choice(string.letters)
    insert_loc = random.randint(0, len(x))
    x = x[:insert_loc] + insert + x[insert_loc:]
    return x

def character_delete(x):
    insert_loc = random.randint(0, len(x)-1)
    x = x[:insert_loc] + x[insert_loc+1:]
    return x

def character_swap(x):
    insert = random.choice(string.letters)
    insert_loc = random.randint(0, len(x)-1)
    x = x[:insert_loc] + insert + x[insert_loc+1:]
    return x

def generate_augs(x):
    functions = [ character_swap, character_delete, character_insert]
    augs = set([random.choice(functions)(copy.copy(x)) for i in range(num_edits_per_line)])
    augs = list(augs)
    while len(augs) < num_edits_per_line:
        while len(augs) < num_edits_per_line:
            augs.append(random.choice(functions)(copy.copy(x)))
            augs = set(augs)
            augs = list(augs)

    return augs

with open(args.base_sentences_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        with open(args.out_folder + "/edits_%d.txt" % i, 'w' ) as f2:
            augs = generate_augs(line)
            for aug in augs:
                f2.write(aug+"\n")
