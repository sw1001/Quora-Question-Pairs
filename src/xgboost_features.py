"""
File: xgboost_features.py
Author: Zhuoran Wu
Email: zw118@georgetown.edu
Reference: https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky/code

Import Train Features and Test Features from: https://www.kaggle.com/c/quora-question-pairs/discussion/31284
"""

import argparse
import functools
from collections import defaultdict

import numpy as np
import pandas as pd

import re
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import xgboost as xgb

WNL = WordNetLemmatizer()


def cutter(word):
    if len(word) < 4:
        return word
    return WNL.lemmatize(WNL.lemmatize(word, "n"), "v")


def clean_process(text):
    if pd.isnull(text):
        return ''

    stops = set(stopwords.words("english"))
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    text = text.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    text = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', text)
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    text = ' '.join([cutter(w) for w in text.split()])

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()

    text = text.split()
    text = [w for w in text if w not in stops]
    text = ' '.join(text)

    # Return a list of words
    return text


def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    r = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return r


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return len(wic) / len(uw)


def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1']) * 1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len(
        [x for x in set(row['question2']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops]) * 1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(
        ''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    r = np.sum(shared_weights) / np.sum(total_weights)
    return r


def build_features(data, stops, weights):
    x = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    x['word_match'] = data.apply(f, axis=1, raw=True)  # 1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    x['tfidf_wm'] = data.apply(f, axis=1, raw=True)  # 2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    x['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True)  # 3

    x['jaccard'] = data.apply(jaccard, axis=1, raw=True)  # 4
    x['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)  # 5
    x['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)  # 6
    x['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)  # 7
    x['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)  # 8

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    x['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True)  # 9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    x['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True)  # 10

    x['same_start'] = data.apply(same_start_word, axis=1, raw=True)  # 11
    x['char_diff'] = data.apply(char_diff, axis=1, raw=True)  # 12

    f = functools.partial(char_diff_unique_stop, stops=stops)
    x['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True)  # 13

    #     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    x['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  # 15

    f = functools.partial(total_unq_words_stop, stops=stops)
    x['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  # 16

    x['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)  # 17

    return x


def main():
    parser = argparse.ArgumentParser(description='XGB with Handcrafted Features')
    parser.add_argument('--save', type=str, default='xgboost_features',
                        help='save_file_names')
    args = parser.parse_args()

    nltk.download('stopwords')
    nltk.download("wordnet")

    df_train = pd.read_csv('../input/train_features.csv', encoding="ISO-8859-1")
    x_train_ab = df_train.iloc[:, 2:-1]
    x_train_ab = x_train_ab.drop('euclidean_distance', axis=1)
    x_train_ab = x_train_ab.drop('jaccard_distance', axis=1)

    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    # Clean the data

    df_train["question1"] = df_train["question1"].fillna("").apply(clean_process)
    df_train["question2"] = df_train["question2"].fillna("").apply(clean_process)
    df_test["question1"] = df_test["question1"].fillna("").apply(clean_process)
    df_test["question2"] = df_test["question2"].fillna("").apply(clean_process)

    ques = pd.concat([df_train[['question1', 'question2']],
                      df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_freq(row):
        return len(q_dict[row['question1']])

    def q2_freq(row):
        return len(q_dict[row['question2']])

    def q1_q2_intersect(row):
        return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))

    df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
    df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
    df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

    df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
    df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
    df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

    test_leaky = df_test.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]
    del df_test

    train_leaky = df_train.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]

    # explore
    stops = set(stopwords.words("english"))

    df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
    df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    print('Building Features')
    x_train = build_features(df_train, stops, weights)
    x_train = pd.concat((x_train, x_train_ab, train_leaky), axis=1)
    y_train = df_train['is_duplicate'].values

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4242)

    # UPDownSampling
    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]
    x_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
    y_train = np.array(
        [0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    pos_valid = x_valid[y_valid == 1]
    neg_valid = x_valid[y_valid == 0]
    x_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_valid = np.array(
        [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    print(np.mean(y_valid))
    del pos_valid, neg_valid

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.02,
        'max_depth': 7,
        'subsample': 0.6,
        'base_score': 0.2
    }

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
    # bst.save_model(args.save + '.mdl')

    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('xgboost_features.mdl')  # load data

    print(log_loss(y_valid, bst.predict(d_valid)))

    print('Building Test Features')
    df_test = pd.read_csv('../input/test_features.csv', encoding="ISO-8859-1")
    x_test_ab = df_test.iloc[:, 2:-1]
    x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
    x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)

    df_test = pd.read_csv('../input/test.csv')
    df_test = df_test.fillna(' ')

    df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
    df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

    x_test = build_features(df_test, stops, weights)
    x_test = pd.concat((x_test, x_test_ab, test_leaky), axis=1)

    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('submission_' + args.save + '.csv', index=False)


if __name__ == '__main__':
    main()
