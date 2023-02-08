import jieba
import re
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer
import pickle
from tqdm import tqdm
import random
from gensim.models import Word2Vec
from utils.tokenizer import MyTokenizer

RANDOM_SEED = 22
torch.manual_seed(RANDOM_SEED)


def load_embedding(embedding_path="code/gensim_train/word2vec.model"):
    model = Word2Vec.load(embedding_path)


def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords("code/utils/stopword.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub("[0-9\.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)


def text_cleaner2(text):
    # 替换html特殊字符
    if type(text) is float:
        return ""
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    # 换行替换为#, 空格替换为&
    text = text.replace("#", "").replace("$", "").replace("&", "")
    text = text.replace("\n", "").replace(" ", "")

    text = jieba.lcut(text)
    # text = list(text)
    return " ".join(text)


class LaicDataset(Dataset):
    def __init__(self, fact, rat_1, rat_2, article, charge, judge):
        self.fact = fact
        self.rat_1 = rat_1
        self.rat_2 = rat_2
        self.article = torch.LongTensor(article)
        self.charge = torch.LongTensor(charge)
        self.judge = torch.Tensor(judge)

    def __getitem__(self, idx):
        return {"fact":
                {
                    "input_ids": self.fact["input_ids"][idx],
                    "token_type_ids": self.fact["token_type_ids"][idx],
                    "attention_mask": self.fact["attention_mask"][idx],
                },
                "rat_1":  {
                    "input_ids": self.rat_1["input_ids"][idx],
                    "token_type_ids": self.rat_1["token_type_ids"][idx],
                    "attention_mask": self.rat_1["attention_mask"][idx],
                },
                "rat_2":   {
                    "input_ids": self.rat_2["input_ids"][idx],
                    "token_type_ids": self.rat_2["token_type_ids"][idx],
                    "attention_mask": self.rat_2["attention_mask"][idx],
                },
                "charge": self.charge[idx],
                "article": self.article[idx], "judge": self.judge[idx]}

    def __len__(self):
        return len(self.judge)


def get_split_dataset(idx, fact, rat_1, rat_2, article, charge, judge):
    fact_cur = {
        "input_ids": fact["input_ids"][idx],
        "token_type_ids": fact["token_type_ids"][idx],
        "attention_mask": fact["attention_mask"][idx],
    }
    rat_1_cur = {
        "input_ids": rat_1["input_ids"][idx],
        "token_type_ids": rat_1["token_type_ids"][idx],
        "attention_mask": rat_1["attention_mask"][idx],
    }
    rat_2_cur = {
        "input_ids": rat_2["input_ids"][idx],
        "token_type_ids": rat_2["token_type_ids"][idx],
        "attention_mask": rat_2["attention_mask"][idx],
    }
    article_cur = pd.Series(article)[idx].tolist()
    charge_cur = pd.Series(charge)[idx].tolist()
    judge_cur = pd.Series(judge)[idx].tolist()
    return LaicDataset(fact_cur, rat_1_cur, rat_2_cur, article_cur, charge_cur, judge_cur)


def load_data(filename, dataset_name, tokenizer, train_set_ratio=0.8, text_clean=True, shuffle=True, maps=None):
    df = pd.read_csv(filename, sep=",")
    path, file = os.path.split(filename)
    stem, suffix = os.path.splitext(file)

    # 读入csv数据，根据上面text_clean参数是否过stopwords
    if text_clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            # df["rat_12"] = ["#"] * len(df)
            df["rat_1"] = ["#"] * len(df)
            df["rat_2"] = ["#"] * len(df)
            for i in tqdm(range(len(df))):
                df["justice"][i] = text_cleaner2(df["justice"][i])
                df["rat_1"][i] = text_cleaner2(df["rat_1_gen"][i])
                df["rat_2"][i] = text_cleaner2(df["rat_2_gen"][i])

            df.to_csv(clean_data_path, index=False, sep=",")
            print("clean data saved: {}".format(clean_data_path))
        df = pd.read_csv(clean_data_path,sep=",")

    if len(df) != len(df.dropna()):
        print("before drop nan, data num: ", len(df))
        print("after drop nan, data num: ", len(df.dropna()))
    df = df.dropna()
    df = df.reset_index()

    # 将fact和opinion文本进行tokenize
    pkl_path = "code/pkl/{}/train_clean.pkl".format(
        dataset_name) if text_clean else "code/pkl/{}/train.pkl".format(dataset_name)
    if not os.path.exists(pkl_path):
        path, _ = os.path.split(pkl_path)
        if not os.path.exists(path):
            os.makedirs(path)

        fact = df["justice"].tolist()
        rat_1 = df["rat_1"].tolist()
        rat_2 = df["rat_2"].tolist()

        fact = tokenizer(fact, max_length=512, return_tensors="pt",padding="max_length",truncation=True)
        rat_1 = tokenizer(rat_1, max_length=128, return_tensors="pt",padding="max_length",truncation=True)
        rat_2 = tokenizer(rat_2, max_length=128, return_tensors="pt",padding="max_length",truncation=True)

        with open(pkl_path, "wb") as f:
            pickle.dump((fact, rat_1, rat_2), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print("pkl data saved: {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        fact, rat_1, rat_2 = pickle.load(f)

    charge = df["charge"].tolist()
    judge = df["judge"].tolist() if "judge" in df.columns else [-1]*len(charge)
    article = df["article"].tolist(
    ) if "article" in df.columns else [-1]*len(charge)

    # 标签转数字id
    def label2idx(label):
        st = set(label)
        lst = sorted(list(st))  # 按照字符串顺序排列
        mp_label2idx, mp_idx2label = dict(), dict()
        for i in range(len(lst)):
            mp_label2idx[lst[i]] = i
            mp_idx2label[i] = lst[i]
        return [mp_label2idx[i] for i in label], mp_label2idx, mp_idx2label

    ret_maps = {}
    if maps is not None:
        print(maps["charge2idx"])
        charge = [maps["charge2idx"][i] for i in charge]
        article = [maps["article2idx"][i] for i in article]
        ret_maps = maps
    else:
        charge, mp_charge2idx, mp_idx2charge = label2idx(charge)
        article, mp_article2idx, mp_idx2article = label2idx(article)

        ret_maps["charge2idx"] = mp_charge2idx
        ret_maps["idx2charge"] = mp_idx2charge
        ret_maps["article2idx"] = mp_article2idx
        ret_maps["idx2article"] = mp_idx2article

    # 划分trainset, validset
    data_split_dir = "data/data_split/{}/".format(dataset_name)
    if not os.path.exists(data_split_dir):
        os.mkdir(data_split_dir)

    tot_size = len(df)
    train_size = int(tot_size*train_set_ratio)
    valid_size = int((tot_size-train_size)/2)
    test_size = tot_size-train_size-valid_size
    random.seed(RANDOM_SEED)
    shuffle_idx = list(range(len(charge)))
    if shuffle:
        random.shuffle(shuffle_idx)
    train_idx, valid_idx, test_idx = shuffle_idx[:train_size], shuffle_idx[
        train_size:train_size+valid_size], shuffle_idx[train_size+valid_size:]
    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    train_df.to_csv(data_split_dir+"train.csv", index=False, sep=",")
    valid_df.to_csv(data_split_dir+"valid.csv", index=False, sep=",")
    test_df.to_csv(data_split_dir+"test.csv", index=False, sep=",")

    trainset = get_split_dataset(
        train_idx, fact, rat_1, rat_2, article, charge, judge)
    validset = get_split_dataset(
        valid_idx, fact, rat_1, rat_2, article, charge, judge)
    testset = get_split_dataset(
        test_idx, fact, rat_1, rat_2, article, charge, judge)

    # dataset = LaicDataset(fact, opinion, rationale, article, charge, judge)
    # trainset, validset, testset = torch.utils.data.random_split(
    #     dataset, [train_size, valid_size, test_size])

    return trainset, validset, testset, ret_maps
