import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from utils.tokenizer import MyTokenizer

from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import numpy as np
import os
from models import (TextCNN, Transformer, LSTM,
                    LSTM_rat_att2, TextCNN_rat_att2,Transformer_rat_att2)
from utils import loader
import argparse
import time
from utils.loss import log_distance_accuracy_function, log_square_loss, acc25

RANDOM_SEED = 22
torch.manual_seed(RANDOM_SEED)

name2model = {
    "TextCNN": TextCNN,
    "LSTM": LSTM,
    "Transformer":Transformer,
    "LSTM_rat_att2": LSTM_rat_att2,
    "TextCNN_rat_att2": TextCNN_rat_att2,
    "Transformer_rat_att2":Transformer_rat_att2,
}

name2tokenizer = {
    "TextCNN": MyTokenizer(embedding_path="code/gensim_train/char2vec.model"),
    "LSTM": MyTokenizer(),
    "Transformer":MyTokenizer(),

    "LSTM_rat_att2": MyTokenizer(),
    "TextCNN_rat_att2": MyTokenizer(),
    "Transformer_rat_att2": MyTokenizer(),
}

name2dim = {
    "TextCNN": 300,
    "LSTM": 300,
    "Transformer":300,

    "LSTM_rat_att2": 300,
    "TextCNN_rat_att2": 300,
    "Transformer_rat_att2": 300,
}


class Trainer:
    def __init__(self, args):

        self.tokenizer = name2tokenizer[args.model_name]

        dataset_name = "laic2021_filter_rat_word"
        data_path = "data/{}.csv".format(dataset_name)
        print("当前数据集路径: ", data_path)

        self.trainset, self.validset, self.testset, self.maps = loader.load_data(
            data_path, dataset_name, self.tokenizer, train_set_ratio=0.8, text_clean=True)

        self.batch = 512
        self.epoch = 100
        self.seq_len = 512
        self.hid_dim = 256
        self.emb_dim = name2dim[args.model_name]


        self.train_dataloader = DataLoader(dataset=self.trainset,
                                           batch_size=self.batch,
                                           shuffle=True,
                                           drop_last=False)

        self.valid_dataloader = DataLoader(dataset=self.validset,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           drop_last=False)

        self.model = name2model[args.model_name](
            vocab_size=self.tokenizer.vocab_size, emb_dim=self.emb_dim, hid_dim=self.hid_dim , maps=self.maps)

        self.cur_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        print("time: ", self.cur_time)
        self.model_name = "{}_{}".format(args.model_name, dataset_name)

        self.task_name = ["article", "charge", "judge"]
        # self.task_name = ["charge"]
        print("task: ", self.task_name)

        print(self.model)
        print("train samples: ", len(self.trainset))
        print("valid samples: ", len(self.validset))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.loss_function = {
            "article": nn.CrossEntropyLoss(),
            "charge": nn.CrossEntropyLoss(),
            "judge": log_square_loss
        }

        self.score_function = {
            "article": self.f1_score_micro,
            "charge": self.f1_score_micro,
            "judge": acc25,
        }

        if args.load_path is not None:
            print("--- stage2 ---")
            print("load model path:", args.load_path)
            self.model_name = self.model_name+"_s3"
            checkpoint = torch.load(args.load_path)
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.evaluate(args.load_path, save_result=False)
            self.set_param_trainable(trainable=True)
            # self.model = self.model.module
        print("parameter counts: ", self.count_parameters())

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # self.model = nn.DataParallel(self.model)

    def set_param_trainable(self, trainable):
        for name, param in self.model.named_parameters():
            if "judge" not in name:
                param.requires_grad = trainable

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def f1_score_micro(self, out, label):
        return f1_score(out.cpu().argmax(1), label.cpu(), average='micro')

    def f1_score_macro(self, out, label):
        return f1_score(out.cpu().argmax(1), label.cpu(), average='macro')

    def train(self):
        best_score = 0
        for e in range(self.epoch):
            # train
            print("--- train ---")
            tq = tqdm(self.train_dataloader)
            for data in tq:
                for k in data:
                    if type(data[k]) is dict:
                        for k2 in data[k]:
                            data[k][k2] = data[k][k2].cuda()
                    else:
                        data[k] = data[k].cuda()

                self.optimizer.zero_grad()
                out = self.model(data)

                loss = 0
                for name in self.task_name:
                    cur_loss = self.loss_function[name](out[name], data[name])
                    loss += cur_loss

                score = {}
                for name in self.task_name:
                    score[name] = self.score_function[name](
                        out[name], data[name])

                loss.backward()
                tq.set_postfix(epoch=e, train_loss=np.around(
                    loss.cpu().detach().numpy(), 4))
                self.optimizer.step()

            # valid
            print("--- valid ---")
            valid_out = self.infer(self.model, self.valid_dataloader)

            name = self.task_name[0]
            cur_score = self.score_function[name](valid_out[name]["pred"],
                                                  valid_out[name]["truth"])

            if cur_score > best_score:
                best_score = cur_score
                model_save_dir = "code/logs/{}/{}/".format(
                    self.model_name, self.cur_time)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                save_path = model_save_dir+"best_model.pt"
                print("best model saved!")
                torch.save(
                    {"model": self.model, "optimizer": self.optimizer}, save_path)

            # test
            self.evaluate(save_path, save_result=False,
                          evaluate_self_testset=True)

    def infer_one(self, text):
        tokens = self.tokenizer(text, return_tensors="pt",
                                padding="max_length", max_length=768, truncation=True)

    def infer(self, model, data_loader):
        tq = tqdm(data_loader)
        eval_out = {k: [] for k in self.task_name}
        for data in tq:
            for k in data:
                if type(data[k]) is dict:
                    for k2 in data[k]:
                        data[k][k2] = data[k][k2].cuda()
                else:
                    data[k] = data[k].cuda()

            with torch.no_grad():
                out = model(data)
                for name in self.task_name:
                    eval_out[name].append((out[name], data[name]))

        for name in eval_out.keys():
            pred = torch.cat([i[0] for i in eval_out[name]])
            truth = torch.cat([i[1] for i in eval_out[name]])
            eval_out[name] = {"pred": pred, "truth": truth}

        for name in self.task_name:
            if name in ["article", "charge"]:
                print("{} micro f1:".format(name), self.f1_score_micro(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
                print("{} macro f1:".format(name), self.f1_score_macro(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
            elif name in ["judge"]:
                print("judge log distance:", log_distance_accuracy_function(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
                print("judge acc25:", acc25(
                    eval_out[name]["pred"], eval_out[name]["truth"]))
        return eval_out

    def evaluate(self, load_path, save_result, print_confusing_matrix = True, evaluate_self_testset=False):
        """
            load_path: saved model path
            save_result: True or False. save result to csv file
        """
        print("--- evaluate on testset: ---")

        if evaluate_self_testset:
            testset = self.testset
        else:
            dataset_name = "laic2021_filter_pgn_test_v3_2dec"
            # dataset_name = "laic2021_filter_pgn_test_v2"
            # dataset_name = "laic2021_filter_c3vg"
            # dataset_name = "laic2021_filter_bart_2dec"
            # dataset_name = "laic2021_filter_bart_2model"
            dataset_name = "cail_test_samples"
            data_path = "data/{}.csv".format(dataset_name)
            print("当前数据集路径: ", data_path)
            testset, _, _, _ = loader.load_data(
                data_path, dataset_name, self.tokenizer, train_set_ratio=1,
                text_clean=True, shuffle=False, maps=self.maps)

        print("test samples: ", len(testset))

        test_dataloader = DataLoader(dataset=testset,
                                     batch_size=self.batch,
                                     shuffle=False,
                                     drop_last=False)

        print("--- test ---")
        print("load model path: ", load_path)
        checkpoint = torch.load(load_path)
        model = checkpoint['model']
        print(model)

        test_out = self.infer(model, test_dataloader)

        if print_confusing_matrix:
            print("charge confusion matrix: ")
            pred = test_out["article"]["pred"].cpu().numpy().argmax(1)
            truth = test_out["article"]["truth"].cpu().numpy()
            con_mat = confusion_matrix(truth, pred)
            np.savetxt("code/res/pred.txt", pred)
            np.savetxt("code/res/truth.txt", truth)
            # np.savetxt(confusion_matrix(truth, pred))

        if save_result:
            df_test = pd.read_csv(data_path)
            df_test["charge_pred"] = [self.maps["idx2charge"][i]
                                      for i in test_out["charge"]["pred"].cpu().numpy().argmax(1)]
            df_test["judge_pred"] = test_out["judge"]["pred"].round().cpu().numpy()
            df_test.to_csv("code/res/result.csv", index=False, sep=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', help='gpu')
    parser.add_argument(
        '--model_name', default='TextCNN_rat_att2', help='model_name')
    parser.add_argument('--load_path', default=None, help='load model path')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainer = Trainer(args)
    trainer.train()


    # eval_path = "code/logs/Transformer_rat_att2_laic2021_filter_rat_word/2022-03-18-11:24:32/best_model.pt"
    # trainer.evaluate(
    #     eval_path,
    #     save_result=False,
    #     print_confusing_matrix=True,
    #     evaluate_self_testset=False
    # )
