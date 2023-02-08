import pandas as pd

df = pd.read_csv("data/laic2021_rat_clean.csv")

print(df.columns)

facts, zm_rat, xq_rat = [], [], []
for fact in df["justice"]:
    fact = fact.replace("#", "").replace("&", "")
    facts.append(fact)

for rat1 in df["rat_1"]:
    rat1 = rat1.replace("#", "").replace("&", "")
    zm_rat.append(rat1)

for rat2 in df["rat_2"]:
    rat2 = rat2.replace("#", "").replace("&", "")
    xq_rat.append(rat2)

with open("train.src", "w") as f:
    for line in facts:
        f.write(line+"\n")

with open("train_zm.tgt", "w") as f:
    for line in zm_rat:
        f.write(line+"\n")

with open("train_xq.tgt", "w") as f:
    for line in xq_rat:
        f.write(line+"\n")
