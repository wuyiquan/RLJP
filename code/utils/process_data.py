import re
import pandas as pd
from tqdm import tqdm


def prev_deal(data_0):
    if '-' in data_0:
        data_0_abs = data_0[1:]
    else:
        data_0_abs = data_0
    digit_range = re.match('(\d+)', data_0_abs)
    if len(digit_range.group(1)) > 12:
        return '仅支持转换正负万亿范围内的数'
    if '.' in data_0_abs:
        add = data_0_abs.find('.')
        data_0_d = data_0_abs[:add]
        data_0_f = data_0_abs[add:]
        result = digit_cutting(arabic2chinese(data_0_d)) + \
            arabic2chinese(data_0_f)
    else:
        result = digit_cutting(arabic2chinese(data_0_abs))
    if '-' in data_0:
        return '负' + result
    else:
        return result


def arabic2chinese(data_1):

    dir = {'1': '一', '2': '二',
           '3': '三', '4': '四',
           '5': '五', '6': '六',
           '7': '七', '8': '八',
           '9': '九', '0': '零',
           '.': '点'
           }
    return ''.join(dir[ch] for ch in data_1)


def digit_cutting(data_2):
    '''对不同长度的数进行切割'''
    length = len(data_2)
    if length == 1:
        pass
    elif 1 < length <= 4:
        data_2 = unit_insert(data_2)
        if data_2[0] == '零':
            data_2 = data_2[1:]
    elif 4 < length <= 8:
        last_four = data_2[-4:]
        prev_four = data_2[:-4]
        if len(prev_four) != 1:
            m = unit_insert(prev_four)
        else:
            m = prev_four
        data_2 = m + '万' + unit_insert(last_four)
    elif 8 < length <= 12:
        last_four = data_2[-4:]
        mid_four = data_2[-8:-4]
        prev_four = data_2[:-8]
        if len(prev_four) == 1:
            m = prev_four
        else:
            m = unit_insert(prev_four)
        if unit_insert(mid_four) == '' and unit_insert(last_four) != '':
            data_2 = m + '零' + unit_insert(last_four)
        elif unit_insert(mid_four) == '' and unit_insert(last_four) == '':
            data_2 = m + '亿'
        else:
            data_2 = m + '亿' + \
                unit_insert(mid_four) + '万' + unit_insert(last_four)
    if data_2[:2] == '一十':
        data_2 = data_2[1:]
    return data_2


def unit_insert(data_3):
    string = '十百千'
    length = len(data_3)
    data_3_list = list(data_3)
    for i in range(length - 1):
        data_3_list.insert(-(2*i+1), string[i])
    data_3 = ''.join(data_3_list)
    for i in ['零千', '零百', '零十', '零零零', '零零']:
        k = data_3.find(i)
        if k != -1:
            data_3 = re.sub(i, '零', data_3)
    if data_3[-1] == '零':
        return data_3[:-1]
    else:
        return data_3


def getlaws(_seg):
    laws = []
    _seg = _seg.replace(" ", "")
    segments1 = _seg.split('。')
    for seg in segments1:
        if '《中华人民' in seg or '依照《' in seg or '按照《' in seg or '依据《' in seg:
            sentences = seg
            sentences = sentences.replace('以及', '，')
            sentences = sentences.replace('条《', '条，《')
            sentences = sentences.replace('款《', '款，《')
            sentences = sentences.replace('、第《', '、《')
            sentences = sentences.replace('及', '，')
            sentences = sentences.replace('共和国', 'ghg')
            sentences = sentences.replace('和', '，')
            sentences = sentences.replace('ghg', '共和国')
            sentences = sentences.replace('的规定', '')
            sentences = sentences.replace('之规定', '')
            sentences = sentences.split('，')
            for sen in sentences:
                sentences1 = sen.split('、')
                for sen1 in sentences1:
                    falv = re.findall(r"《(.+?)》", sen1)
                    if len(falv) != 0:
                        current_law = falv[0]
                    if '第' in sen1:
                        try:
                            temp_dict = {}
                            temp_tiao = re.findall(r"第(.+?)条", sen1)
                            if len(temp_tiao) == 0:
                                temp_tiao = ["零"]
                            laws.append(temp_tiao)
                        except:
                            pass

            return laws
    return laws


def ar2cn(num):
    num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五',
                '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
    index_dict = {1: '', 2: '十', 3: '百', 4: '千',
                  5: '万', 6: '十', 7: '百', 8: '千', 9: '亿'}

    nums = list(num)
    nums_index = [x for x in range(1, len(nums)+1)][-1::-1]

    string = ''
    for index, item in enumerate(nums):
        string = "".join(
            (string, num_dict[item], index_dict[nums_index[index]]))

    string = re.sub("零[十百千零]*", "零", string)
    string = re.sub("零万", "万", string)
    string = re.sub("亿万", "亿零", string)
    string = re.sub("零零", "零", string)
    string = re.sub("零\\b", "", string)
    return string


def get_rationales(df):
    rationales = []

    for opinion in df["opinion"]:
        sent = opinion.split("。")[0]
        rationales.append(sent)
    res = []
    for text in rationales:
        text = text.split("，")
        rationale = []
        for sent in text:
            if "罪" in sent:
                break
            rationale.append(sent)
        res.append("，".join(rationale))
    return res


df = pd.read_csv("data/trainset.csv", sep=",")
rationales = get_rationales(df)
df["rationale"] = rationales

articles = []
for i in tqdm(range(len(df["opinion"]))):
    opinion = df["opinion"][i]
    res = getlaws(opinion)
    res = sorted(res, key=lambda i: len(i), reverse=True)

    text = ""
    if len(res) > 0:
        text = res[0][0]
        text = text.replace("第", "")
    else:
        text = "###"
    if str.isdigit(text):  # 如果是数字，就从阿拉伯转中文
        text = ar2cn(text)
    articles.append(text)

articles = pd.Series(articles)
article_lst = articles.value_counts()[:150]
article_lst = article_lst.drop(["###", "零"])


drop_idx = []
df["article"] = articles
for i in range(len(articles)):
    cur_article = articles[i]
    if cur_article not in article_lst:
        drop_idx.append(i)


df = df.drop(index=drop_idx, axis=0)  # len: 27617
df = df[df["article"].str.len() >= 4]
df = df[df["justice"].str.len() >= 75]
df = df[df["rationale"].str.len() >= 15]

print(len(df))
print(df.columns.tolist())
df.to_csv("data/trainset_tot.csv", index=False, sep=",")
