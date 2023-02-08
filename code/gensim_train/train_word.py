import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re
from tqdm import tqdm
import random
from gensim.models import Word2Vec
import time
import json
import jieba


def laic_data():
    testset = pd.read_csv(
        "data/laic2021.csv")
    fact = testset["justice"].tolist()
    opinion = testset["opinion"].tolist()
    data = []
    for line in tqdm(fact):
        data.append(jieba.lcut(line.replace("\n", "")))
    for line in tqdm(opinion):
        data.append(jieba.lcut(line.replace("\n", "")))
    # fact.extend(opinion)
    return data


def cail_data(file):
    fact = []
    with open(file) as f:
        for line in tqdm(f.readlines()):
            json_obj = json.loads(line)
            # 字向量
            # fact.append(list(json_obj["fact"]))
            line = json_obj["fact"]
            line = line.replace("\n", "")
            fact.append(jieba.lcut(line))
            # break
    return fact


def save_cut_corpus():
    fact = laic_data()
    train_file = "data/cail/final_all_data/first_stage/train.json"
    test_file = "data/cail/final_all_data/first_stage/test.json"

    with open("word_cut_corpus_laic.txt","w") as f:
        for line in fact:
            f.write(" ".join(line)+"\n")


def get_cut_corpus():
    corpus = []
    data_dir = "code/gensim_train/"
    data_dir = ""
    with open(data_dir+"word_cut_corpus_laic.txt", "r") as f:
        for line in tqdm(f.readlines()):
            line = line.replace("\n", "")
            line = line.split()
            corpus.append(line)
    with open(data_dir + "word_cut_corpus_cail_train.txt", "r") as f:
        for line in tqdm(f.readlines()):
            line = line.replace("\n", "")
            line = line.split()
            corpus.append(line)
    with open(data_dir + "word_cut_corpus_cail_test.txt", "r") as f:
        for line in tqdm(f.readlines()):
            line = line.replace("\n", "")
            line = line.split()
            corpus.append(line)
    return corpus


# save_cut_corpus()

corpus = get_cut_corpus()

print(len(corpus))
# corpus = corpus[:1000]

start_time = time.time()
model = Word2Vec(corpus, vector_size=300, max_final_vocab=21000, workers=4)
end_time = time.time()
print("cost time: ", end_time - start_time)

model.save("word2vec.model")
