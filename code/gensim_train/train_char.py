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
import pickle


def laic_data():
    testset = pd.read_csv(
        "data/laic2021.csv")
    fact = testset["justice"].tolist()
    opinion = testset["opinion"].tolist()
#     word_set = set(c for line in fact for c in line)
    fact = [list(line) for line in fact]
    opinion = [list(line) for line in opinion]
    fact.extend(opinion)
    return fact


def cail_data(file):
    fact = []
    with open(file) as f:
        for line in tqdm(f.readlines()):
            json_obj = json.loads(line)
            fact.append(list(json_obj["fact"]))
#             break
    return fact


def train_word2vec():
    fact = laic_data()
    train_file = "data/cail/final_all_data/first_stage/train.json"
    test_file = "data/cail/final_all_data/first_stage/test.json"

    fact_2 = cail_data(train_file)
    fact_3 = cail_data(test_file)

    fact_tot = []
    fact_tot.extend(fact)
    fact_tot.extend(fact_2)
    fact_tot.extend(fact_3)

    word_set = set(c for line in fact_tot for c in line)
    print("word set: ", len(word_set))

    sentences = fact_tot

    start_time = time.time()
    model = Word2Vec(sentences, vector_size = 300, min_count=1, workers=4)
    end_time = time.time()
    print("cost time: ",end_time - start_time)

    model.save("word2vec.model")

def load_word2vec(file_path):
    model = Word2Vec.load("code/gensim_train/word2vec.model")
    a = 1


load_word2vec("word2vec.model")

# with open("gensim_model.pkl", "wb") as f:
#     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)