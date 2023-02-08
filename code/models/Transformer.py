import torch.nn as nn
import torch
from utils.tokenizer import MyTokenizer

class Transformer(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim

        self.tokenizer = MyTokenizer()
        vectors = self.tokenizer.load_embedding()
        vectors = torch.Tensor(vectors)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(vectors)

        self.transformer_enc = nn.TransformerEncoderLayer(self.emb_dim, nhead=10,batch_first=True)

        self.fc_article = nn.Linear(self.emb_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.emb_dim, self.charge_class_num)
        self.fc_judge = nn.Linear(self.emb_dim, 1)

        self.dropout = nn.Dropout(0.4)

    def forward(self, data):
        text = data["fact"]["input_ids"].cuda()
        x = self.embedding(text)
        out = self.transformer_enc(x)
        out = self.dropout(out)

        out = torch.mean(out,dim=1)

        out_charge = self.fc_charge(out)
        out_article = self.fc_article(out)
        out_judge = self.fc_judge(out)
        return {
            "article": out_article,
            "charge": out_charge,
            "judge": out_judge
        }
