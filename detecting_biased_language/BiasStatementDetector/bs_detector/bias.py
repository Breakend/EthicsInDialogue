#!/usr/bin/python
# coding: utf-8
"""
Created on June 04, 2015
@author: C.J. Hutto
"""

import nltk
#from vaderSentiment.vaderSentiment import sentiment as vader_sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as vader_sentiment
from pattern.en import parse, Sentence, parse, modality, mood
from pattern.en import sentiment as pattern_sentiment
from textstat.textstat import textstat

import pprint as pp
import sys
import csv
import numpy as np
import pyprind
import re
import argparse


def get_list_from_file(file_name):
    with open(file_name, "r") as f1:
        l = f1.read().lower().split('\n')
    return l


def append_to_file(file_name, line):
    # ...append a line of text to a file
    with open(file_name, 'a') as f1:
        f1.write(line)
        f1.write("\n")


def count_feature_list_freq(feat_list, words, bigrams, trigrams):
    cnt = 0
    words = []
    for w in words:
        if w in feat_list:
            cnt += 1
            words.append(w)
    for b in bigrams:
        if b in feat_list:
            cnt += 1
            words.append(b)
    for t in trigrams:
        if t in feat_list:
            cnt += 1
            words.append(t)
    return cnt, words


def count_liwc_list_freq(liwc_list, words_list):
    cnt = 0
    words = []
    for w in words_list:
        if w in liwc_list:
            cnt += 1
            words.append(w)
        for lw in liwc_list:
            if str(lw).endswith('*') and str(w).startswith(lw):
                cnt += 1
                if w not in words:
                    words.append(w)
    return cnt, words


##### List of assertive verbs and factive verbs extracted from:
# Joan B. Hooper. 1975. On assertive predicates. In J. Kimball, editor,
# Syntax and Semantics, volume 4, pages 91–124. Academic Press, New York.
#########################################################################
assertives = get_list_from_file('../ref_lexicons/ref_assertive_verbs')
factives = get_list_from_file('../ref_lexicons/ref_factive_verbs')

##### List of hedges extracted from:
# Ken Hyland. 2005. Metadiscourse: Exploring Interaction in Writing.
# Continuum, London and New York.
#########################################################################
hedges = get_list_from_file('../ref_lexicons/ref_hedge_words')

##### List of implicative verbs extracted from:
# Lauri Karttunen. 1971. Implicative verbs. Language, 47(2):340–358.
#########################################################################
implicatives = get_list_from_file('../ref_lexicons/ref_implicative_verbs')

##### List of strong/weak subjective words extracted from:
# Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
# Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005,
# Vancouver, Canada.
#########################################################################
subj_strong = get_list_from_file('../ref_lexicons/ref_subj_strong')
subj_weak = get_list_from_file('../ref_lexicons/ref_subj_weak')

##### List of bias words extracted from:
# Marta Recasens, Cristian Danescu-Niculescu-Mizil, and Dan
# Jurafsky. 2013. Linguistic Models for Analyzing and Detecting Biased
# Language. Proceedings of ACL 2013.
#########################################################################
biased = get_list_from_file('../ref_lexicons/ref_bias_words')

##### List of coherence markers extracted from:
# Knott, Alistair. 1996. A Data-Driven Methodology for Motivating a Set of
# Coherence Relations. Ph.D. dissertation, University of Edinburgh, UK.
# Note: probably could be cleaned up a lot... e.g., ...
# import re
# def find_whole_word(w):
#    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search
# lst = sorted(get_list_from_file('ref_coherence_markers'))
# for w in lst:
#    excl = [i for i in lst if i != w]
#    for i in excl:
#        if find_whole_word(w)(i):
#            print w, "-->", i
#########################################################################
coherence = get_list_from_file('../ref_lexicons/ref_coherence_markers')

##### List of degree modifiers and opinion words extracted from:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
#  Sentiment Analysis of Social Media Text. Eighth International Conference on
#  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
#########################################################################
modifiers = get_list_from_file('../ref_lexicons/ref_degree_modifiers')
opinionLaden = get_list_from_file('../ref_lexicons/ref_vader_words')
vader_sentiment_analysis = vader_sentiment()

##### Lists of LIWC category words
# liwc 3rd person pronoun count (combines S/he and They)
liwc_3pp = ["he", "hed", "he'd", "her", "hers", "herself", "hes", "he's", "him", "himself", "his", "oneself",
            "she", "she'd", "she'll", "shes", "she's", "their*", "them", "themselves", "they", "theyd",
            "they'd", "theyll", "they'll", "theyve", "they've"]
# liwc auxiliary verb count
liwc_aux = ["aint", "ain't", "am", "are", "arent", "aren't", "be", "became", "become", "becomes",
            "becoming", "been", "being", "can", "cannot", "cant", "can't", "could", "couldnt",
            "couldn't", "couldve", "could've", "did", "didnt", "didn't", "do", "does", "doesnt",
            "doesn't", "doing", "done", "dont", "don't", "had", "hadnt", "hadn't", "has", "hasnt",
            "hasn't", "have", "havent", "haven't", "having", "hed", "he'd", "heres", "here's",
            "hes", "he's", "id", "i'd", "i'll", "im", "i'm", "is", "isnt", "isn't", "itd", "it'd",
            "itll", "it'll", "it's", "ive", "i've", "let", "may", "might", "mightve", "might've",
            "must", "mustnt", "must'nt", "mustn't", "mustve", "must've", "ought", "oughta",
            "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "shall", "shant", "shan't",
            "she'd", "she'll", "shes", "she's", "should", "shouldnt", "should'nt", "shouldn't",
            "shouldve", "should've", "thatd", "that'd", "thatll", "that'll", "thats", "that's",
            "theres", "there's", "theyd", "they'd", "theyll", "they'll", "theyre", "they're",
            "theyve", "they've", "was", "wasnt", "wasn't", "we'd", "we'll", "were", "weren't",
            "weve", "we've", "whats", "what's", "wheres", "where's", "whod", "who'd", "wholl",
            "who'll", "will", "wont", "won't", "would", "wouldnt", "wouldn't", "wouldve", "would've",
            "youd", "you'd", "youll", "you'll", "youre", "you're", "youve", "you've"]
# liwc adverb count
liwc_adv = ["about", "absolutely", "actually", "again", "also", "anyway*", "anywhere", "apparently",
            "around", "back", "basically", "beyond", "clearly", "completely", "constantly", "definitely",
            "especially", "even", "eventually", "ever", "frequently", "generally", "here", "heres", "here's",
            "hopefully", "how", "however", "immediately", "instead", "just", "lately", "maybe", "mostly",
            "nearly", "now", "often", "only", "perhaps", "primarily", "probably", "push*", "quick*", "rarely",
            "rather", "really", "seriously", "simply", "so", "somehow", "soon", "sooo*", "still", "such",
            "there", "theres", "there's", "tho", "though", "too", "totally", "truly", "usually", "very", "well",
            "when", "whenever", "where", "yet"]
# liwc preposition count
liwc_prep = ["about", "above", "across", "after", "against", "ahead", "along", "among*", "around", "as", "at",
             "atop", "away", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond",
             "by", "despite", "down", "during", "except", "for", "from", "in", "inside", "insides", "into", "near",
             "of", "off", "on", "onto", "out", "outside", "over", "plus", "since", "than", "through*", "thru", "til",
             "till", "to", "toward*", "under", "underneath", "unless", "until", "unto", "up", "upon", "wanna", "with",
             "within", "without"]
# liwc conjunction count
liwc_conj = ["also", "although", "and", "as", "altho", "because", "but", "cuz", "how", "however", "if", "nor",
             "or", "otherwise", "plus", "so", "then", "tho", "though", "til", "till", "unless", "until", "when",
             "whenever", "whereas", "whether", "while"]
# liwc discrepency word count
liwc_discr = ["besides", "could", "couldnt", "couldn't", "couldve", "could've", "desir*", "expect*", "hope", "hoped",
              "hopeful", "hopefully",
              "hopefulness", "hopes", "hoping", "ideal*", "if", "impossib*", "inadequa*", "lack*", "liabilit*",
              "mistak*", "must", "mustnt",
              "must'nt", "mustn't", "mustve", "must've", "need", "needed", "needing", "neednt", "need'nt", "needn't",
              "needs", "normal", "ought",
              "oughta", "oughtnt", "ought'nt", "oughtn't", "oughtve", "ought've", "outstanding", "prefer*", "problem*",
              "rather", "regardless",
              "regret*", "should", "shouldnt", "should'nt", "shouldn't", "shoulds", "shouldve", "should've",
              "undesire*", "undo", "unneccess*",
              "unneed*", "unwant*", "wanna", "want", "wanted", "wanting", "wants", "wish", "wished", "wishes",
              "wishing", "would", "wouldnt",
              "wouldn't", "wouldve", "would've", "yearn*"]
# liwc tentative word count
liwc_tent = ["allot", "almost", "alot", "ambigu*", "any", "anybod*", "anyhow", "anyone*", "anything", "anytime",
             "anywhere",
             "apparently", "appear", "appeared", "appearing", "appears", "approximat*", "arbitrar*", "assum*", "barely",
             "bet",
             "bets", "betting", "blur*", "borderline*", "chance", "confus*", "contingen*", "depend", "depended",
             "depending",
             "depends", "disorient*", "doubt*", "dubious*", "dunno", "fairly", "fuzz*", "generally", "guess", "guessed",
             "guesses",
             "guessing", "halfass*", "hardly", "hazie*", "hazy", "hesita*", "hope", "hoped", "hopeful", "hopefully",
             "hopefulness",
             "hopes", "hoping", "hypothes*", "hypothetic*", "if", "incomplet*", "indecis*", "indefinit*", "indetermin*",
             "indirect*",
             "kind(of)", "kinda", "kindof", "likel*", "lot", "lotof", "lots", "lotsa", "lotta", "luck", "lucked",
             "lucki*", "luckless*",
             "lucks", "lucky", "mainly", "marginal*", "may", "maybe", "might", "mightve", "might've", "most", "mostly",
             "myster*", "nearly",
             "obscur*", "occasional*", "often", "opinion", "option", "or", "overall", "partly", "perhaps", "possib*",
             "practically", "pretty",
             "probable", "probablistic*", "probably", "puzzl*", "question*", "quite", "random*", "seem", "seemed",
             "seeming*", "seems", "shaki*",
             "shaky", "some", "somebod*", "somehow", "someone*", "something*", "sometime", "sometimes", "somewhat",
             "sort", "sorta", "sortof",
             "sorts", "sortsa", "spose", "suppose", "supposed", "supposes", "supposing", "supposition*", "tempora*",
             "tentativ*", "theor*",
             "typically", "uncertain*", "unclear*", "undecided*", "undetermin*", "unknow*", "unlikel*", "unluck*",
             "unresolv*", "unsettl*",
             "unsure*", "usually", "vague*", "variab*", "varies", "vary", "wonder", "wondered", "wondering", "wonders"]
# liwc certainty word count
liwc_cert = ["absolute", "absolutely", "accura*", "all", "altogether", "always", "apparent", "assur*", "blatant*",
             "certain*", "clear", "clearly",
             "commit", "commitment*", "commits", "committ*", "complete", "completed", "completely", "completes",
             "confidence", "confident",
             "confidently", "correct*", "defined", "definite", "definitely", "definitive*", "directly", "distinct*",
             "entire*", "essential",
             "ever", "every", "everybod*", "everything*", "evident*", "exact*", "explicit*", "extremely", "fact",
             "facts", "factual*", "forever",
             "frankly", "fundamental", "fundamentalis*", "fundamentally", "fundamentals", "guarant*", "implicit*",
             "indeed", "inevitab*",
             "infallib*", "invariab*", "irrefu*", "must", "mustnt", "must'nt", "mustn't", "mustve", "must've",
             "necessar*", "never", "obvious*",
             "perfect*", "positiv*", "precis*", "proof", "prove*", "pure*", "sure*", "total", "totally", "true",
             "truest", "truly", "truth*",
             "unambigu*", "undeniab*", "undoubt*", "unquestion*", "wholly"]
# liwc causation word count
liwc_causn = ["activat*", "affect", "affected", "affecting", "affects", "aggravat*", "allow*", "attribut*", "based",
              "bases", "basis",
              "because", "boss*", "caus*", "change", "changed", "changes", "changing", "compel*", "compliance",
              "complie*", "comply*",
              "conclud*", "consequen*", "control*", "cos", "coz", "create*", "creati*", "cuz", "deduc*", "depend",
              "depended", "depending",
              "depends", "effect*", "elicit*", "experiment", "force*", "foundation*", "founded", "founder*",
              "generate*", "generating",
              "generator*", "hence", "how", "hows", "how's", "ignit*", "implica*", "implie*", "imply*", "inact*",
              "independ*", "induc*",
              "infer", "inferr*", "infers", "influenc*", "intend*", "intent*", "justif*", "launch*", "lead*", "led",
              "made", "make", "maker*",
              "makes", "making", "manipul*", "misle*", "motiv*", "obedien*", "obey*", "origin", "originat*", "origins",
              "outcome*", "permit*",
              "pick ", "produc*", "provoc*", "provok*", "purpose*", "rational*", "react*", "reason*", "response",
              "result*", "root*", "since",
              "solution*", "solve", "solved", "solves", "solving", "source*", "stimul*", "therefor*", "thus",
              "trigger*", "use", "used", "uses",
              "using", "why"]
# liwc work word count
liwc_work = ["absent*", "academ*", "accomplish*", "achiev*", "administrat*", "advertising", "advis*", "agent", "agents",
             "ambiti*", "applicant*",
             "applicat*", "apprentic*", "assign*", "assistan*", "associat*", "auditorium*", "award*", "beaten",
             "benefits", "biolog*", "biz",
             "blackboard*", "bldg*", "book*", "boss*", "broker*", "bureau*", "burnout*", "business*", "busy",
             "cafeteria*", "calculus", "campus*",
             "career*", "ceo*", "certif*", "chairm*", "chalk", "challeng*", "champ*", "class", "classes", "classmate*",
             "classroom*", "collab*",
             "colleague*", "colleg*", "com", "commerc*", "commute*", "commuting", "companies", "company", "comput*",
             "conferenc*", "conglom*",
             "consult*", "consumer*", "contracts", "corp", "corporat*", "corps", "counc*", "couns*", "course*",
             "coworker*", "credential*",
             "credit*", "cubicle*", "curricul*", "customer*", "cv*", "deadline*", "dean*", "delegat*", "demote*",
             "department*", "dept", "desk*",
             "diplom*", "director*", "dissertat*", "dividend*", "doctor*", "dorm*", "dotcom", "downsiz*", "dropout*",
             "duti*", "duty", "earn*",
             "econ*", "edit*", "educat*", "elementary", "employ*", "esl", "exam", "exams", "excel*", "executive*",
             "expel*", "expulsion*",
             "factories", "factory", "facult*", "fail*", "fax*", "feedback", "finaliz*", "finals", "financ*", "fired",
             "firing", "franchis*",
             "frat", "fratern*", "freshm*", "gmat", "goal*", "gov", "govern*", "gpa", "grad", "grade*", "grading",
             "graduat*", "gre", "hardwork*",
             "headhunter*", "highschool*", "hire*", "hiring", "homework*", "inc", "income*", "incorp*", "industr*",
             "instruct*", "interview*",
             "inventory", "jd", "job*", "junior*", "keyboard*", "kinderg*", "labor*", "labour*", "laidoff", "laptop*",
             "lawyer*", "layoff*",
             "lead*", "learn*", "lectur*", "legal*", "librar*", "lsat", "ltd", "mailroom*", "majoring", "majors",
             "manag*", "manufact*", "market*",
             "masters", "math*", "mcat", "mda", "meeting*", "memo", "memos", "menial", "mentor*", "merger*", "mfg",
             "mfr", "mgmt", "mgr", "midterm*",
             "motiv*", "negotiat*", "ngo", "nonprofit*", "occupa*", "office*", "org", "organiz*", "outlin*",
             "outsourc*", "overpaid", "overtime",
             "overworked", "paper*", "pay*", "pc*", "pen", "pencil*", "pens", "pension*", "phd*", "photocop*", "pledg*",
             "police", "policy",
             "political", "politics", "practice", "prereq*", "presentation*", "presiden*", "procrastin*", "produc*",
             "prof", "profession*",
             "professor*", "profit*", "profs", "program*", "project", "projector*", "projects", "prom", "promot*",
             "psych", "psychol*", "publish",
             "qualifi*", "quiz*", "read", "recruit*", "register*", "registra*", "report*", "requir*", "research*",
             "resource", "resources",
             "resourcing", "responsib*", "resume", "retire*", "retiring", "review*", "rhetor*", "salar*", "scholar",
             "scholaring", "scholarly",
             "scholars", "scholarship*", "scholastic*", "school*", "scien*", "secretar*", "sector*", "semester*",
             "senior*", "servic*",
             "session*", "sickday*", "sickleave*", "sophom*", "sororit*", "staff*", "stapl*", "stipend*", "stock",
             "stocked", "stocker",
             "stocks", "student*", "studied", "studies", "studious", "study*", "succeed*", "success*", "supervis*",
             "syllabus*", "taught", "tax",
             "taxa*", "taxed", "taxes", "taxing", "teach*", "team*", "tenure*", "test", "tested", "testing", "tests",
             "textbook*", "theses",
             "thesis", "toefl", "trade*", "trading", "transcript*", "transfer*", "tutor*", "type*", "typing",
             "undergrad*", "underpaid",
             "unemploy*", "universit*", "unproduc*", "upperclass*", "varsit*", "vita", "vitas", "vocation*", "vp*",
             "wage", "wages", "warehous*",
             "welfare", "work ", "workabl*", "worked", "worker*", "working*", "works", "xerox*"]
# liwc achievement word count
liwc_achiev = ["abilit*", "able*", "accomplish*", "ace", "achiev*", "acquir*", "acquisition*", "adequa*", "advanc*",
               "advantag*", "ahead",
               "ambiti*", "approv*", "attain*", "attempt*", "authorit*", "award*", "beat", "beaten", "best", "better",
               "bonus*", "burnout*",
               "capab*", "celebrat*", "challeng*", "champ*", "climb*", "closure", "compet*", "conclud*", "conclus*",
               "confidence", "confident",
               "confidently", "conquer*", "conscientious*", "control*", "create*", "creati*", "crown*", "defeat*",
               "determina*", "determined",
               "diligen*", "domina*", "domote*", "driven", "dropout*", "earn*", "effect*", "efficien*", "effort*",
               "elit*", "enabl*", "endeav*",
               "excel*", "fail*", "finaliz*", "first", "firsts", "founded", "founder*", "founding", "fulfill*", "gain*",
               "goal*", "hero*", "honor*",
               "honour*", "ideal*", "importan*", "improve*", "improving", "inadequa*", "incapab*", "incentive*",
               "incompeten*", "ineffect*",
               "initiat*", "irresponsible*", "king*", "lazie*", "lazy", "lead*", "lesson*", "limit*", "lose", "loser*",
               "loses", "losing", "loss*",
               "lost", "master", "mastered", "masterful*", "mastering", "mastermind*", "masters", "mastery", "medal*",
               "mediocr*", "motiv*",
               "obtain*", "opportun*", "organiz*", "originat*", "outcome*", "overcome", "overconfiden*", "overtak*",
               "perfect*", "perform*",
               "persever*", "persist*", "plan", "planned", "planner*", "planning", "plans", "potential*", "power*",
               "practice", "prais*",
               "presiden*", "pride", "prize*", "produc*", "proficien*", "progress", "promot*", "proud*", "purpose*",
               "queen", "queenly", "quit",
               "quitt*", "rank", "ranked", "ranking", "ranks", "recover*", "requir*", "resolv*", "resourceful*",
               "responsib*", "reward*", "skill",
               "skilled", "skills", "solution*", "solve", "solved", "solves", "solving", "strateg*", "strength*",
               "striv*", "strong*", "succeed*",
               "success*", "super", "superb*", "surviv*", "team*", "top", "tried", "tries", "triumph*", "try", "trying",
               "unable", "unbeat*",
               "unproduc*", "unsuccessful*", "victor*", "win", "winn*", "wins", "won", "work ", "workabl*", "worked",
               "worker*", "working*", "works"]


def extract_bias_features(text):
    features = {}
    txt_lwr = str(text).lower()
    words = nltk.word_tokenize(txt_lwr)
    words = [w for w in words if len(w) > 0 and w not in '.?!,;:\'s"$']
    if len(words) < 1:
        return None
    unigrams = sorted(list(set(words)))
    bigram_tokens = nltk.bigrams(words)
    bigrams = [" ".join([w1, w2]) for w1, w2 in sorted(set(bigram_tokens))]
    trigram_tokens = nltk.trigrams(words)
    trigrams = [" ".join([w1, w2, w3]) for w1, w2, w3 in sorted(set(trigram_tokens))]
    # print words
    # print unigrams
    # print bigrams
    # print trigrams
    # print "----------------------"

    # word count
    features['word_count'] = float(len(words))

    # unique word count
    features['unique_word_count'] = float(len(unigrams))

    # coherence marker count
    count, instances = count_feature_list_freq(coherence, words, bigrams, trigrams)
    # if count > 0:
    features['coherence_marker_count'] = count
    features['coherence_marker_prop'] = round(float(count) / float(len(words)), 4)
    features['coherence_marker_list'] = instances

    # degree modifier count
    count, instances = count_feature_list_freq(modifiers, words, bigrams, trigrams)
    #if count > 0:
    features['degree_modifier_count'] = count
    features['degree_modifier_prop'] = round(float(count) / float(len(words)), 4)
    features['degree_modifier_list'] = instances

    # hedge word count
    count, instances = count_feature_list_freq(hedges, words, bigrams, trigrams)
    #if count > 0:
    features['hedge_word_count'] = count
    features['hedge_word_prop'] = round(float(count) / float(len(words)), 4)
    features['hedge_word_list'] = instances

    # factive verb count
    count, instances = count_feature_list_freq(factives, words, bigrams, trigrams)
    #if count > 0:
    features['factive_verb_count'] = count
    features['factive_verb_prop'] = round(float(count) / float(len(words)), 4)
    features['factive_verb_list'] = instances

    # assertive verb count
    count, instances = count_feature_list_freq(assertives, words, bigrams, trigrams)
    #if count > 0:
    features['assertive_verb_count'] = count
    features['assertive_verb_prop'] = round(float(count) / float(len(words)), 4)
    features['assertive_verb_list'] = instances

    # implicative verb count
    count, instances = count_feature_list_freq(implicatives, words, bigrams, trigrams)
    #if count > 0:
    features['implicative_verb_count'] = count
    features['implicative_verb_prop'] = round(float(count) / float(len(words)), 4)
    features['implicative_verb_list'] = instances

    # bias words and phrases count
    count, instances = count_feature_list_freq(biased, words, bigrams, trigrams)
    #if count > 0:
    features['bias_count'] = count
    features['bias_prop'] = round(float(count) / float(len(words)), 4)
    features['bias_list'] = instances

    # opinion word count
    count, instances = count_feature_list_freq(opinionLaden, words, bigrams, trigrams)
    #if count > 0:
    features['opinion_count'] = count
    features['opinion_prop'] = round(float(count) / float(len(words)), 4)
    features['opinion_list'] = instances

    # weak subjective word count
    count, instances = count_feature_list_freq(subj_weak, words, bigrams, trigrams)
    #if count > 0:
    features['subjective_weak_count'] = count
    features['subjective_weak_prop'] = round(float(count) / float(len(words)), 4)
    features['subjective_weak_list'] = instances

    # strong subjective word count
    count, instances = count_feature_list_freq(subj_strong, words, bigrams, trigrams)
    #if count > 0:
    features['subjective_strong_count'] = count
    features['subjective_strong_prop'] = round(float(count) / float(len(words)), 4)
    features['subjective_strong_list'] = instances

    # composite sentiment score using VADER sentiment analysis package
    compound_sentiment = vader_sentiment_analysis.polarity_scores(text)['compound']
    features['vader_composite_sentiment'] = float(compound_sentiment)

    # subjectivity score using Pattern.en
    pattern_subjectivity = pattern_sentiment(text)[1]
    features['subjectivity_score'] = round(pattern_subjectivity, 4)

    # modality (certainty) score and mood using  http://www.clips.ua.ac.be/pages/pattern-en#modality
    sentence = parse(text, lemmata=True)
    sentenceObj = Sentence(sentence)
    features['modality'] = round(modality(sentenceObj), 4)
    try:
        features['mood'] = mood(sentenceObj)
    except IndexError as e:
        print "IndexError: %s" % e
        print "Ignoring..."
        features['mood'] = 'err'

    # Flesch-Kincaid Grade Level (reading difficulty) using textstat
    try:
        features['flesch-kincaid_grade_level'] = float(textstat.flesch_kincaid_grade(text))
    except TypeError as e:
        print "TypeError: %s" % e
        print "Ignoring..."
        features['flesch-kincaid_grade_level'] = 0.0

    # liwc 3rd person pronoun count (combines S/he and They)
    count, instances = count_liwc_list_freq(liwc_3pp, words)
    #if count > 0:
    features['liwc_3rd_person_pronoum_count'] = count
    features['liwc_3rd_person_pronoun_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_3rd_person_pronoun_list'] = instances

    # liwc auxiliary verb count
    count, instances = count_liwc_list_freq(liwc_aux, words)
    #if count > 0:
    features['liwc_auxiliary_verb_count'] = count
    features['liwc_auxiliary_verb_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_auxiliary_verb_list'] = instances

    # liwc adverb count
    count, instances = count_liwc_list_freq(liwc_adv, words)
    #if count > 0:
    features['liwc_adverb_count'] = count
    features['liwc_adverb_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_adverb_list'] = instances

    # liwc preposition count
    count, instances = count_liwc_list_freq(liwc_prep, words)
    #if count > 0:
    features['liwc_preposition_count'] = count
    features['liwc_preposition_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_preposition_list'] = instances

    # liwc conjunction count
    count, instances = count_liwc_list_freq(liwc_conj, words)
    #if count > 0:
    features['liwc_conjunction_count'] = count
    features['liwc_conjunction_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_conjunction_list'] = instances

    # liwc discrepency word count
    count, instances = count_liwc_list_freq(liwc_discr, words)
    #if count > 0:
    features['liwc_discrepency_word_count'] = count
    features['liwc_discrepency_word_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_discrepency_word_list'] = instances

    # liwc tentative word count
    count, instances = count_liwc_list_freq(liwc_tent, words)
    #if count > 0:
    features['liwc_tentative_word_count'] = count
    features['liwc_tentative_word_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_tentative_word_list'] = instances

    # liwc certainty word count
    count, instances = count_liwc_list_freq(liwc_cert, words)
    #if count > 0:
    features['liwc_certainty_word_count'] = count
    features['liwc_certainty_word_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_certainty_word_list'] = instances

    # liwc causation word count
    count, instances = count_liwc_list_freq(liwc_causn, words)
    #if count > 0:
    features['liwc_causation_word_count'] = count
    features['liwc_causation_word_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_causation_word_list'] = instances

    # liwc work word count
    count, instances = count_liwc_list_freq(liwc_work, words)
    #if count > 0:
    features['liwc_work_word_count'] = count
    features['liwc_work_word_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_work_word_list'] = instances

    # liwc achievement word count
    count, instances = count_liwc_list_freq(liwc_achiev, words)
    #if count > 0:
    features['liwc_achievement_word_count'] = count
    features['liwc_achievement_word_prop'] = round(float(count) / float(len(words)), 4)
    features['liwc_achievement_word_list'] = instances

    return features


def get_raw_data_for_features(list_of_sentences, KEYS_DONE=False):
    data = []
    # bar = pyprind.ProgBar(len(list_of_sentences), monitor=False, stream=sys.stdout)  # show a progression bar on the screen
    # print ""
    for s in list_of_sentences:
        s = s.replace('</s> ', '')\
            .replace('<first_speaker> ', '')\
            .replace('<second_speaker> ', '')\
            .replace('<third_speaker>', '')\
            .replace('<at> ', '')\
            .replace('</d> ', '')\
            .replace(' </s>', '')\
            .replace(' </d>', '')\
            .replace('__eou__ ', '')\
            .replace(' __eou__', '')\
            .replace(';', '')\
            .replace('__eot__ ', '')
        s = re.sub('<speaker_[0-9]+> ', '', s)
        s = unicode(s, errors='ignore')
        if len(s) > 3:
            feat = extract_bias_features(s)
            if feat is None:  # if no features were found (s was too small)
                continue      # continue to next sentence in the loop
            bias = compute_bias(s, feat)
            # Get the count and prop keys, not the list of instances
            keys = sorted(filter(lambda k: not k.endswith('_list'), feat.keys()))
            # if keys not printed yet, print *all* of them, and add the *filtered* ones to data
            if not KEYS_DONE:
                # print feat.keys()
                print ['sentence', 'bias_score']+keys
                data.append(['sentence', 'bias_score']+keys)
                KEYS_DONE = True
            # print *all* values
            # print feat.values()+[bias]
            # but add *filtered* values
            data.append([s, bias]+[feat[k] for k in keys])
            # write '.' with no space and no new line -- just like c function print()
            #sys.stdout.write('.')
            #sys.stdout.flush()
        # bar.update()
    return data
#print_raw_data_for_features(get_list_from_file('input_text_original'))

def compute_bias(sentence_text, features=None):
    if features is None:
        features = extract_bias_features(unicode(sentence_text, errors='ignore'))
    BS_SCORE = (-0.5581467 +
          0.3477007   * features.get('vader_composite_sentiment', 0.0) +
          -2.0461103  * features.get('opinion_prop', 0.0) +
          0.5164345   * features.get('modality', 0.0) +
          8.3551389   * features.get('liwc_3rd_person_pronoun_prop', 0.0) +
          4.5965115   * features.get('liwc_tentative_word_prop', 0.0) +
          5.737545    * features.get('liwc_achievement_word_prop', 0.0) +
          5.6573254   * features.get('liwc_discrepency_word_prop', 0.0) +
          -0.953181   * features.get('bias_prop', 0.0) +
          9.811681    * features.get('liwc_work_word_prop', 0.0) +
          -16.6359498 * features.get('factive_verb_prop', 0.0) +
          3.059548    * features.get('hedge_word_prop', 0.0) +
          -3.5770891  * features.get('assertive_verb_prop', 0.0) +
          5.0959142   * features.get('subjective_strong_prop', 0.0) +
          4.872367    * features.get('subjective_weak_prop', 0.0))
    return BS_SCORE

def demo_sample_news_story_sentences():
    for statement in get_list_from_file('input_text'):
        if len(statement) > 3:
            bias = compute_bias(statement)
            print(statement, bias)

def get_bias(data_name, data_paths, debug=False, batch_size=10000):
    print "loading data..."
    lines = []
    for f in data_paths:
        if debug: lines.extend(get_list_from_file(f)[:100])
        else: lines.extend(get_list_from_file(f))
    print "data loaded: %d sentences." % len(lines)

    print "\nget data stats in batches..."
    str_idx = -1
    total, average = [], []
    itt = 0

    # bar = pyprind.ProgBar(len(lines), monitor=False, stream=sys.stdout)  # show a progression bar on the screen
    for start_idx in range(0, len(lines), batch_size):
        stop_idx = min(len(lines), start_idx+batch_size)

        if start_idx == 0:  # first data batch
            data = np.array(get_raw_data_for_features(lines[start_idx:stop_idx], KEYS_DONE=False))
            # save the column index where we have string values:
            str_idx = np.where(data[0]=='mood')[0][0]
            n_rows = len(data) - 1
        else:
            data = np.array(get_raw_data_for_features(lines[start_idx:stop_idx], KEYS_DONE=True))
            n_rows = len(data)
        # print "Got %d sentence stats" % n_rows
    
        # print "Saving to CVS file..."
        with open('%s.csv' % data_name, 'ab') as f:
            writer = csv.writer(f)
            writer.writerows(data)  # write full matrix to file
            itt += 1

            # create a no-string data to sum and average all rows
            data_no_str = np.hstack((
                [[0.]] * n_rows,                       # replace sentence column with 0's
                data[len(data) - n_rows:, 1:str_idx],  # skip 1st line (if feature names) and 1st column (sentences)
                [[0.]] * n_rows,                       # replace mood column with 0's
                data[len(data) - n_rows:, str_idx+1:]  # skip 1st line (if feature names)
            )).astype(np.float)

            total.append( np.sum(data_no_str, axis=0) )
            average.append( np.mean(data_no_str, axis=0) )

            if stop_idx == len(lines):  # last data batch
                total = np.sum(total, axis=0)
                average = np.mean(average, axis=0)
                writer.writerow(total)
                writer.writerow(average)

        print "saved %d lines." % (itt*batch_size),
        # bar.update()

    return total, average


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', choices=['twitter', 'reddit', 'ubuntu', 'movie'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--bs', type=int, default=10000, help='batch size')
    args = parser.parse_args()

    # demo_sample_news_story_sentences()

    if args.data_name == 'twitter':
        data_paths = [
            '/home/ml/nangel3/research/data/twitter/train.txt',
            '/home/ml/nangel3/research/data/twitter/valid.txt',
            '/home/ml/nangel3/research/data/twitter/test.txt'
        ]
    elif args.data_name == 'reddit':
        data_paths = [
            '/home/ml/nangel3/research/data/reddit/allnews/allnews_train.txt',
            '/home/ml/nangel3/research/data/reddit/allnews/allnews_val.txt',
            '/home/ml/nangel3/research/data/reddit/allnews/allnews_test.txt'
        ]
    elif args.data_name == 'ubuntu':
        data_paths = [
            '/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_training_text.txt',
            '/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_valid_text.txt',
            '/home/ml/nangel3/research/data/ubuntu/UbuntuDialogueCorpus/raw_test_text.txt'
        ]
    elif args.data_name == 'movie':
        # extracted raw movie dialogs to
        # '/home/ml/nangel3/research/data/cornell_movie-dialogs_corpus/movie_dialogs.txt'
        data_paths = ['/home/ml/nangel3/research/data/cornell_movie-dialogs_corpus/movie_dialogs.txt']
    else:
        print "ERROR: unrecognized data name"
        data_paths = []

    if args.debug:
        total, average = get_bias(args.data_name, data_paths, args.debug, 10)
    else:
        total, average = get_bias(args.data_name, data_paths, args.debug, args.bs)
    print "=====TOTAL======"
    print total
    print "====AVERAGE====="
    print average
    print "==========="
    with open('%s_recap.csv' % args.data_name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(total)
        writer.writerow(average)

