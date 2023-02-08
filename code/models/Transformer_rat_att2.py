import torch.nn as nn
import torch
from utils.tokenizer import MyTokenizer


class Transformer_rat_att2(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=128, hid_dim=128, maps=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])

        # 加载词向量
        self.tokenizer = MyTokenizer()
        vectors = self.tokenizer.load_embedding()
        vectors = torch.Tensor(vectors)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(vectors)

        bidirectional = True
        self.fact_lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=bidirectional,
                                 batch_first=True, dropout=0.5)
        self.rat_1_lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=bidirectional,
                                  batch_first=True, dropout=0.5)
        self.rat_2_lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=bidirectional,
                                  batch_first=True, dropout=0.5)

        if bidirectional:
            self.hid_dim *= 2
        self.fr1_W = nn.Parameter(torch.zeros(
            self.hid_dim, self.hid_dim))
        self.fr2_W = nn.Parameter(torch.zeros(
            self.hid_dim, self.hid_dim))

        multihead_dim = self.hid_dim
        self.rat_1_multihead_attn = nn.MultiheadAttention(multihead_dim, num_heads=8, batch_first=True)
        self.rat_2_multihead_attn = nn.MultiheadAttention(multihead_dim, num_heads=8, batch_first=True)

        self.fact_enc = nn.TransformerEncoderLayer(self.hid_dim, nhead=8,batch_first=True)
        self.rat_1_enc = nn.TransformerEncoderLayer(self.hid_dim, nhead=8,batch_first=True)
        self.rat_2_enc = nn.TransformerEncoderLayer(self.hid_dim, nhead=8,batch_first=True)

        self.fc_input_dim = self.hid_dim
        self.dropout = nn.Dropout(0.4)
        self.fc_article = nn.Linear(self.fc_input_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.fc_input_dim*2, self.charge_class_num)
        self.fc_judge = nn.Linear(self.fc_input_dim*3, 1)

    def forward(self, data):
        fact = data["fact"]["input_ids"].cuda()
        rat_1 = data["rat_1"]["input_ids"].cuda()
        rat_2 = data["rat_2"]["input_ids"].cuda()

        fact = self.embedding(fact)
        rat_1 = self.embedding(rat_1)
        rat_2 = self.embedding(rat_2)

        fact_hidden, _ = self.fact_lstm(fact)  # [64, 512, 256]
        rat_1_hidden, _ = self.rat_1_lstm(rat_1)  # [64, 256, 256]
        rat_2_hidden, _ = self.rat_2_lstm(rat_2)  # [64, 256, 256]

        def attention(seq_hidden, att_seq_hidden, w):
            # [batch, seq1, dim] x [dim, dim] => [batch, seq1, dim]
            u = torch.matmul(seq_hidden, w)
            # [batch, seq1, dim] x [batch, dim, seq2] => [batch, seq1, seq2]
            v = torch.bmm(u, att_seq_hidden.permute(0, 2, 1))
            alpha = nn.Softmax(dim=1)(v)  # [batch, seq1, seq2]
            # [batch, seq1, seq2] x [batch, seq2, dim] => [batch, seq1, dim]
            encoded = torch.bmm(alpha, att_seq_hidden)
            encoded = encoded.squeeze()
            return encoded

        fact_seq_hidden = fact_hidden
        
        fr1_seq_hidden, _ = self.rat_1_multihead_attn(fact_hidden, rat_1_hidden, rat_1_hidden)
        fr2_seq_hidden, _ = self.rat_2_multihead_attn(fact_hidden, rat_2_hidden, rat_2_hidden)

        fact = self.fact_enc(fact_seq_hidden)
        fr1 = self.rat_1_enc(fr1_seq_hidden)
        fr2 = self.rat_2_enc(fr2_seq_hidden)

        fact = torch.mean(fact, dim=1)
        rat_1 = torch.mean(fr1, dim=1)
        rat_2 = torch.mean(fr2, dim=1)

        input_article = torch.cat([fact], dim=1)
        input_charge = torch.cat([fact, rat_1], dim=1)
        input_judge = torch.cat([fact, rat_1, rat_2], dim=1)

        out_article = self.fc_article(input_article)
        out_charge = self.fc_charge(input_charge)
        out_judge = self.fc_judge(input_judge)
        return {
            "article": out_article,
            "charge": out_charge,
            "judge": out_judge
        }
