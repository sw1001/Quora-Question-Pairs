import feat_gen
import importlib

import pandas as pd
import numpy as np

import pickle
import argparse
import functools
import gensim
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

from collections import defaultdict
from collections import Counter

from nltk.corpus import stopwords
from nltk import word_tokenize

import dask
from dask.dataframe import from_pandas

from tqdm import tqdm, tqdm_pandas

from fuzzywuzzy import fuzz

from gensim.models import KeyedVectors


importlib.reload(feat_gen)

#################################
# Feature 2
#################################


print('loading data.....')
train_orig =  pd.read_csv('../../input/train.csv', header=0)
test_orig =  pd.read_csv('../../input/test.csv', header=0)

train_df = train_orig.copy()
test_df = test_orig.copy()

print('cleaning data.....')
train_df['question1'].fillna('', inplace=True)
train_df['question2'].fillna('', inplace=True)
test_df['question1'].fillna('', inplace=True)
test_df['question2'].fillna('', inplace=True)


train_df['q1_clean1'] = train_df.apply(lambda x: feat_gen.clean1(x['question1']), axis=1)
train_df['q2_clean1'] = train_df.apply(lambda x: feat_gen.clean1(x['question2']), axis=1)
test_df['q1_clean1'] = test_df.apply(lambda x: feat_gen.clean1(x['question1']), axis=1)
test_df['q2_clean1'] = test_df.apply(lambda x: feat_gen.clean1(x['question2']), axis=1)

# Build features ##############################################################
print('magic features.....')
(train_1, test_1) = feat_gen.magic1(train_df, test_df)
train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)
del train_1, test_1

###############################################################################
print('wordmatch features.....')
(train_1, test_1) = feat_gen.wordmatch1(train_df, test_df)
(train_2, test_2) = feat_gen.wordmatch1(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
# Drop redundant columns
test_df = test_df.drop('id',1)

###############################################################################
print('ngram features.........')
(train_1, test_1) = feat_gen.ngram_stats2(train_df, test_df)
(train_2, test_2) = feat_gen.ngram_stats2(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
print('edit_distance....')
(train_1, test_1) = feat_gen.edit_distance(train_df, test_df)
(train_2, test_2) = feat_gen.edit_distance(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
print('fuzzy_feats....')
(train_1, test_1) = feat_gen.fuzzy_feats(train_df, test_df)
(train_2, test_2) = feat_gen.fuzzy_feats(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

print('saving data....')
f = open('train_F2.pickle', 'wb')
pickle.dump(train_df, f)
f.close()

f = open('test_F2.pickle', 'wb')
pickle.dump(test_df, f)
f.close()

#################################
# Feature 3
#################################

print('loading original data.....')
train_orig =  pd.read_csv('../../input/train.csv', header=0)
test_orig =  pd.read_csv('../../input/test.csv', header=0)

print('loading F2 features....')
f = open('train_F2.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F2.pickle', 'rb')
test_df = pickle.load(f)
f.close()

###############################################################################
print('ngram char features.........')
(train_1, test_1) = feat_gen.ngram_stats2(train_df, test_df, append = 'char', char=True)
(train_2, test_2) = feat_gen.ngram_stats2(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='char_c', char=True)

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
print('saving data....')

f = open('train_F3.pickle', 'wb')
pickle.dump(train_df, f)
f.close()

f = open('test_F3.pickle', 'wb')
pickle.dump(test_df, f)
f.close()

#################################
# Feature 4
#################################

print('loading F3 features....')
f = open('train_F3.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F3.pickle', 'rb')
test_df = pickle.load(f)
f.close()

###############################################################################
print('building nieve bayes features')
maxNumFeatures = 30000

# bag of letter sequences (chars)
BagOfWordsExtractor1 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                       analyzer='char', ngram_range=(1, 1),
                                       binary=True, lowercase=True)

BagOfWordsExtractor2 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                       analyzer='char', ngram_range=(2, 2),
                                       binary=True, lowercase=True)

BagOfWordsExtractor3 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                       analyzer='char', ngram_range=(3, 3),
                                       binary=True, lowercase=True)

BagOfWordsExtractor1234 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                          analyzer='char', ngram_range=(1, 4),
                                          binary=True, lowercase=True)

BOW_ExList = [BagOfWordsExtractor1, BagOfWordsExtractor2, BagOfWordsExtractor3, BagOfWordsExtractor1234]
BOW_labels = ['1', '2', '3', '1234']

qcols = ['question1', 'question2']

for i, extr in enumerate(BOW_ExList):
    # Build vectorizors and transform data
    print(i)
    extr.fit(pd.concat((train_df[qcols[0]], train_df[qcols[1]])).unique())
    BOW_q1_chgram = extr.transform(train_df[qcols[0]])
    BOW_q2_chgram = extr.transform(train_df[qcols[1]])
    test_BOW_q1_chgram = extr.transform(test_df[qcols[0]])
    test_BOW_q2_chgram = extr.transform(test_df[qcols[1]])

    # make features
    BPW_chgram_add = BOW_q1_chgram + BOW_q2_chgram
    BPW_chgram_intersec = BOW_q1_chgram.multiply(BOW_q2_chgram).sign()
    test_BPW_chgram_add = test_BOW_q1_chgram + test_BOW_q2_chgram
    test_BPW_chgram_intersec = test_BOW_q1_chgram.multiply(test_BOW_q2_chgram).sign()
    del BOW_q1_chgram, BOW_q2_chgram, test_BOW_q1_chgram, test_BOW_q2_chgram

    # Predict test/training features
    model = MultinomialNB(alpha=1)
    y_data = train_df['is_duplicate'].values

    # Train
    train_df['nBayes' + BOW_labels[i] + '_add'] = cross_val_predict(model, BPW_chgram_add, y_data,
                                                                    method='predict_proba', cv=5)[:, 1]
    train_df['nBayes' + BOW_labels[i] + '_intersec'] = cross_val_predict(model, BPW_chgram_intersec, y_data,
                                                                         method='predict_proba', cv=5)[:, 1]

    # Test
    model.fit(BPW_chgram_add, y_data)
    test_df['nBayes' + BOW_labels[i] + '_add'] = model.predict_proba(test_BPW_chgram_add)[:, 1]

    model.fit(BPW_chgram_intersec, y_data)
    test_df['nBayes' + BOW_labels[i] + '_intersec'] = model.predict_proba(test_BPW_chgram_intersec)[:, 1]

# Cleanup
del BPW_chgram_add, BPW_chgram_intersec, test_BPW_chgram_add, test_BPW_chgram_intersec
del BagOfWordsExtractor1, BagOfWordsExtractor2, BagOfWordsExtractor3, BagOfWordsExtractor1234

# bag of words (words)

BOW_extr_word1 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                 analyzer='word', ngram_range=(1, 3),
                                 binary=True, lowercase=True)

BOW_ExList = [BOW_extr_word1]
BOW_labels = ['1']

qcols = ['q1_clean1', 'q2_clean1']

for i, extr in enumerate(BOW_ExList):
    # Build vectorizors and transform data
    print(i)
    extr.fit(pd.concat((train_df[qcols[0]], train_df[qcols[1]])).unique())
    BOW_q1_chgram = extr.transform(train_df[qcols[0]])
    BOW_q2_chgram = extr.transform(train_df[qcols[1]])
    test_BOW_q1_chgram = extr.transform(test_df[qcols[0]])
    test_BOW_q2_chgram = extr.transform(test_df[qcols[1]])

    # make features
    BPW_chgram_add = BOW_q1_chgram + BOW_q2_chgram
    BPW_chgram_intersec = BOW_q1_chgram.multiply(BOW_q2_chgram).sign()
    test_BPW_chgram_add = test_BOW_q1_chgram + test_BOW_q2_chgram
    test_BPW_chgram_intersec = test_BOW_q1_chgram.multiply(test_BOW_q2_chgram).sign()
    del BOW_q1_chgram, BOW_q2_chgram, test_BOW_q1_chgram, test_BOW_q2_chgram

    # Predict test/training features
    model = MultinomialNB(alpha=1)
    y_data = train_df['is_duplicate'].values

    # Train
    train_df['nBayes_w' + BOW_labels[i] + '_add'] = cross_val_predict(model, BPW_chgram_add, y_data,
                                                                      method='predict_proba', cv=5)[:, 1]
    train_df['nBayes_w' + BOW_labels[i] + '_intersec'] = cross_val_predict(model, BPW_chgram_intersec, y_data,
                                                                           method='predict_proba', cv=5)[:, 1]

    # Test
    model.fit(BPW_chgram_add, y_data)
    test_df['nBayes_w' + BOW_labels[i] + '_add'] = model.predict_proba(test_BPW_chgram_add)[:, 1]

    model.fit(BPW_chgram_intersec, y_data)
    test_df['nBayes_w' + BOW_labels[i] + '_intersec'] = model.predict_proba(test_BPW_chgram_intersec)[:, 1]

# Cleanup
del BPW_chgram_add, BPW_chgram_intersec, test_BPW_chgram_add, test_BPW_chgram_intersec
del BOW_extr_word1

cols = ['nBayes1_add', 'nBayes2_add', 'nBayes3_add',
        'nBayes1_intersec', 'nBayes2_intersec', 'nBayes3_intersec',
        'nBayes1234_add', 'nBayes1234_intersec',
        'nBayes_w1_add', 'nBayes_w1_intersec', 'is_duplicate']

train_df[cols].corr()

print('saving data....')

f = open('train_F4.pickle', 'wb')
pickle.dump(train_df, f)
f.close()

f = open('test_F4.pickle', 'wb')
pickle.dump(test_df, f)
f.close()


#################################
# Feature 5
#################################

stop_words = stopwords.words('english')

# Load and clean data #########################################################
# print('loading original data.....')
train_df = pd.read_csv('../../input/train.csv', header=0)
test_df = pd.read_csv('../../input/test.csv', header=0)

"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


###############################################################################
# Train
print('length features....')
train_df['len_char_q1'] = train_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_char_q2'] = train_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_word_q1'] = train_df.question1.apply(lambda x: len(str(x).split()))
train_df['len_word_q2'] = train_df.question2.apply(lambda x: len(str(x).split()))

# Test
test_df['len_char_q1'] = test_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_char_q2'] = test_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_word_q1'] = test_df.question1.apply(lambda x: len(str(x).split()))
test_df['len_word_q2'] = test_df.question2.apply(lambda x: len(str(x).split()))

###############################################################################
# Try paralell computation with dask
# Train
print('extra fuzzy features, train....')
train_dd = from_pandas(train_df[['question1', 'question2']], npartitions=8)

start_time = time.time()
train_df['fuzz_qratio'] = train_dd.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1,
                                         meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
train_df['fuzz_WRatio'] = train_dd.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1,
                                         meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
train_df['fuzz_token_set_ratio'] = train_dd.apply(
    lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1,
    meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
train_df['fuzz_token_sort_ratio'] = train_dd.apply(
    lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1,
    meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
print((time.time() - start_time))
del train_dd

# Test
print('extra fuzzy features, test....')
test_dd = from_pandas(test_df[['question1', 'question2']], npartitions=8)

start_time = time.time()
test_df['fuzz_qratio'] = test_dd.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1,
                                       meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
test_df['fuzz_WRatio'] = test_dd.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1,
                                       meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
test_df['fuzz_token_set_ratio'] = test_dd.apply(
    lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1,
    meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
test_df['fuzz_token_sort_ratio'] = test_dd.apply(
    lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1,
    meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
print((time.time() - start_time))
del test_dd
###############################################################################

model = gensim.models.KeyedVectors.load_word2vec_format('../../input/GoogleNews-vectors-negative300.bin.gz',
                                                        binary=True)

tqdm_pandas(tqdm(desc="Train wmd:", total=len(train_df)))
train_df['wmd'] = train_df.progress_apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
tqdm_pandas(tqdm(desc="Test wmd:", total=len(test_df)))
test_df['wmd'] = test_df.progress_apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
del model

norm_model = gensim.models.KeyedVectors.load_word2vec_format('../../input/GoogleNews-vectors-negative300.bin.gz',
                                                             binary=True)
norm_model.init_sims(replace=True)
tqdm_pandas(tqdm(desc="Train norm wmd:", total=len(train_df)))
train_df['norm_wmd'] = train_df.progress_apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
tqdm_pandas(tqdm(desc="Test norm wmd:", total=len(test_df)))
test_df['norm_wmd'] = test_df.progress_apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
del norm_model

###############################################################################
# word2Vec features
# Train set
model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz',
                                                        binary=True)
question1_vectors = np.zeros((train_df.shape[0], 300))
question2_vectors = np.zeros((train_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(train_df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors = np.zeros((train_df.shape[0], 300))
for i, q in tqdm(enumerate(train_df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

train_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

train_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]

train_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

train_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

train_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]

train_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                       np.nan_to_num(question2_vectors))]

train_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                      np.nan_to_num(question2_vectors))]

train_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
train_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
train_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
train_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

# Test set

question1_vectors = np.zeros((test_df.shape[0], 300))
question2_vectors = np.zeros((test_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(test_df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors = np.zeros((test_df.shape[0], 300))
for i, q in tqdm(enumerate(test_df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

test_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                             np.nan_to_num(question2_vectors))]

test_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

test_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                               np.nan_to_num(question2_vectors))]

test_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                 np.nan_to_num(question2_vectors))]

test_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

test_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                      np.nan_to_num(question2_vectors))]

test_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                     np.nan_to_num(question2_vectors))]

test_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
test_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
test_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
test_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

print('saving data....')

f = open('train_F5_tmp.pickle', 'wb')
pickle.dump(train_df, f)
f.close()

f = open('test_F5_tmp.pickle', 'wb')
pickle.dump(test_df, f)
f.close()

# Clean features
# Train
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.loc[:, 'wmd'].fillna(train_df['wmd'].mean(), inplace=True)
train_df.loc[:, 'norm_wmd'].fillna(train_df['norm_wmd'].mean(), inplace=True)
train_df.loc[:, 'cosine_distance'].fillna(train_df['cosine_distance'].mean(), inplace=True)
train_df.loc[:, 'jaccard_distance'].fillna(train_df['jaccard_distance'].mean(), inplace=True)
train_df.loc[:, 'braycurtis_distance'].fillna(train_df['braycurtis_distance'].mean(), inplace=True)

# Test
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.loc[:, 'wmd'].fillna(test_df['wmd'].mean(), inplace=True)
test_df.loc[:, 'norm_wmd'].fillna(test_df['norm_wmd'].mean(), inplace=True)
test_df.loc[:, 'cosine_distance'].fillna(test_df['cosine_distance'].mean(), inplace=True)
test_df.loc[:, 'jaccard_distance'].fillna(test_df['jaccard_distance'].mean(), inplace=True)
test_df.loc[:, 'braycurtis_distance'].fillna(test_df['braycurtis_distance'].mean(), inplace=True)

f = open('train_F5_tmp_clean.pickle', 'wb')
pickle.dump(train_df, f)
f.close()

f = open('test_F5_tmp_clean.pickle', 'wb')
pickle.dump(test_df, f)
f.close()

################################# Join dataframes to complete feature set #####

f = open('train_F4.pickle', 'rb')
train_F4 = pickle.load(f)
f.close()

f = open('test_F4.pickle', 'rb')
test_F4 = pickle.load(f)
f.close()

f = open('train_F5_tmp_clean.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F5_tmp_clean.pickle', 'rb')
test_df = pickle.load(f)
f.close()

new_feats = ['len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2',
             'fuzz_qratio', 'fuzz_WRatio', 'fuzz_token_set_ratio',
             'fuzz_token_sort_ratio', 'wmd', 'norm_wmd', 'cosine_distance',
             'cityblock_distance', 'jaccard_distance', 'canberra_distance',
             'euclidean_distance', 'minkowski_distance', 'braycurtis_distance',
             'skew_q1vec', 'skew_q2vec', 'kur_q1vec', 'kur_q2vec']

train_F4 = train_F4.combine_first(train_df[new_feats])
test_F4 = test_F4.combine_first(test_df[new_feats])

f = open('train_F5.pickle', 'wb')
pickle.dump(train_F4, f)
f.close()

f = open('test_F5.pickle', 'wb')
pickle.dump(test_F4, f, protocol=4)
f.close()

#################################
# Feature 6
#################################

train_orig = pd.read_csv('../../input/train.csv', header=0)
test_orig = pd.read_csv('../../input/test.csv', header=0)

ques = pd.concat([train_orig[['question1', 'question2']], \
                  test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')
ques.shape

q_dict = defaultdict(set)

for i in range(len(ques)):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])


def q1_q2_intersect(row):
    return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

f = open('train_F5.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F5.pickle', 'rb')
test_df = pickle.load(f)
f.close()

train_df['q1_q2_intersect'] = train_orig['q1_q2_intersect']
test_df['q1_q2_intersect'] = test_orig['q1_q2_intersect']

f = open('train_F6.pickle', 'wb')
pickle.dump(train_df, f)
f.close()

f = open('test_F6.pickle', 'wb')
pickle.dump(test_df, f, protocol=4)
f.close()

#################################
# Feature 7 Skip
#################################

#################################
# Feature 8
#################################

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
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))


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

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True)  # 1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True)  # 2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True)  # 3

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True)  # 4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)  # 5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)  # 6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)  # 7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)  # 8

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True)  # 9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True)  # 10

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True)  # 11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True)  # 12

    f = functools.partial(char_diff_unique_stop, stops=stops)
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True)  # 13

    #     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  # 15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  # 16

    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)  # 17

    return X


parser = argparse.ArgumentParser(description='XGB with Handcrafted Features')
parser.add_argument('--save', type=str, default='XGB_leaky',
                    help='save_file_names')
args = parser.parse_args()

df_train = pd.read_csv('../../input/train_features.csv', encoding="ISO-8859-1")
X_train_ab = df_train.iloc[:, 2:-1]
X_train_ab = X_train_ab.drop('euclidean_distance', axis=1)
X_train_ab = X_train_ab.drop('jaccard_distance', axis=1)

df_train = pd.read_csv('../../input/train.csv')
df_train = df_train.fillna(' ')

df_test = pd.read_csv('../../input/test.csv')
del df_test

# explore
stops = set(stopwords.words("english"))

df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

words = [x for y in train_qs for x in y]
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Building Features')
X_train = build_features(df_train, stops, weights)
X_train = pd.concat((X_train, X_train_ab), axis=1)

train_df = X_train
train_df.fillna(train_df.mean(), inplace=True)

# Test features
print('Building Test Features')
df_test = pd.read_csv('../../input/test_features.csv', encoding="ISO-8859-1")
x_test_ab = df_test.iloc[:, 2:-1]
x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)
df_test = pd.read_csv('../../input/test.csv')
df_test = df_test.fillna(' ')

df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

x_test = build_features(df_test, stops, weights)
x_test = pd.concat((x_test, x_test_ab), axis=1)

test_df = x_test
test_df.fillna(train_df.mean(), inplace=True)

del x_test, X_train, X_train_ab, counts, df_test, df_train, train_qs, weights, x_test_ab
del words

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(train_df.mean(), inplace=True)

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.fillna(train_df.mean(), inplace=True)

dupe_cols = ['word_match', 'len_q1', 'len_q2', 'diff_len', 'fuzz_qratio',
             'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
             'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio'
                                                                      'wmd', 'norm_wmd', 'cosine_distance',
             'cityblock_distance', 'canberra_distance',
             'minkowski_distance', 'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
             'fuzz_token_sort_ratio', 'wmd', 'jaccard']

unique_cols = [x for x in train_df.columns if x not in dupe_cols]

train_df = train_df[unique_cols]
test_df = test_df[unique_cols]

f = open('train_F6.pickle', 'rb')
train_df2 = pickle.load(f)
f.close()

f = open('test_F6.pickle', 'rb')
test_df2 = pickle.load(f)
f.close()

train_df2 = train_df2.combine_first(train_df)
test_df2 = test_df2.combine_first(test_df)

del train_df, test_df

f = open('train_F8.pickle', 'wb')
pickle.dump(train_df2, f)
f.close()

f = open('test_F8.pickle', 'wb')
pickle.dump(test_df2, f, protocol=4)
f.close()

#################################
# Feature 9
#################################

stop_words = stopwords.words('english')

# Load and clean data #########################################################
# print('loading original data.....')
# train_orig =  pd.read_csv('../../input/train.csv', header=0)
# test_orig =  pd.read_csv('../../input/test.csv', header=0)

train_df = pd.read_csv('../../input/train.csv', header=0)
test_df = pd.read_csv('../../input/test.csv', header=0)

train_df.fillna(' ', inplace=True)
test_df.fillna(' ', inplace=True)

# Clean data
train_df['q1_clean'] = train_df['question1'].apply(feat_gen.clean1)
train_df['q2_clean'] = train_df['question2'].apply(feat_gen.clean1)

# Comment out
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec('./word2Vec_models/glove.840B.300d.txt', './word2Vec_models/glove_w2vec.txt')

# from gensim.models import Word2Vec

model = KeyedVectors.load_word2vec_format('./word2Vec_models/glove_w2vec.txt')


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


question1_vectors = np.zeros((train_df.shape[0], 300))
question2_vectors = np.zeros((train_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(train_df.q1_clean.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors = np.zeros((train_df.shape[0], 300))
for i, q in tqdm(enumerate(train_df.q2_clean.values)):
    question2_vectors[i, :] = sent2vec(q)

train_df['cosine_distance2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                               np.nan_to_num(question2_vectors))]

train_df['cityblock_distance2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                     np.nan_to_num(question2_vectors))]

train_df['jaccard_distance2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                 np.nan_to_num(question2_vectors))]

train_df['canberra_distance2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

train_df['euclidean_distance2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                     np.nan_to_num(question2_vectors))]

train_df['minkowski_distance2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                        np.nan_to_num(question2_vectors))]

train_df['braycurtis_distance2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                       np.nan_to_num(question2_vectors))]

train_df['skew_q1vec2'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
train_df['skew_q2vec2'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
train_df['kur_q1vec2'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
train_df['kur_q2vec2'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

test_df['q1_clean'] = test_df['question1'].apply(feat_gen.clean1)
test_df['q2_clean'] = test_df['question2'].apply(feat_gen.clean1)

# Test set
# model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
question1_vectors = np.zeros((test_df.shape[0], 300))
question2_vectors = np.zeros((test_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(test_df.q1_clean.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors = np.zeros((test_df.shape[0], 300))
for i, q in tqdm(enumerate(test_df.q2_clean.values)):
    question2_vectors[i, :] = sent2vec(q)

test_df['cosine_distance2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

test_df['cityblock_distance2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]

test_df['jaccard_distance2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

test_df['canberra_distance2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

test_df['euclidean_distance2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]

test_df['minkowski_distance2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                       np.nan_to_num(question2_vectors))]

test_df['braycurtis_distance2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                      np.nan_to_num(question2_vectors))]

test_df['skew_q1vec2'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
test_df['skew_q2vec2'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
test_df['kur_q1vec2'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
test_df['kur_q2vec2'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

feat_names = ['cosine_distance2', 'cityblock_distance2', 'jaccard_distance2',
              'canberra_distance2', 'euclidean_distance2', 'minkowski_distance2',
              'braycurtis_distance2', 'skew_q1vec2', 'skew_q2vec2',
              'kur_q1vec2', 'kur_q2vec2']

del question1_vectors, question2_vectors

f = open('train_F8.pickle', 'rb')
train_df2 = pickle.load(f)
f.close()

f = open('test_F8.pickle', 'rb')
test_df2 = pickle.load(f)
f.close()

train_df2 = pd.concat([train_df2, train_df[feat_names]], axis=1)
test_df2 = pd.concat([test_df2, test_df[feat_names]], axis=1)

# del train_df, test_df


train_df2.to_csv('train_F9.csv')
test_df2.to_csv('test_F9.csv')
