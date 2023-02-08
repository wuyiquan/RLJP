import re
import pandas as pd
from tqdm import tqdm
from utils.cn2ar import chinese2arabic


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


if __name__=="__main__":

    df = pd.read_csv("code/res/result.csv", sep=",")
    articles = []
    single_article_stat = []
    wrong = 0
    wrongcase = []
    for i in tqdm(range(len(df["opinion"]))):
        opinion = df["opinion"][i]
        res = getlaws(opinion)
        lst = []
        for arti in res:
            if arti[0] == '零':
                continue
            lst.append(arti[0])
        single_article_stat += lst
        lst.sort(key=lambda x: len(x), reverse=True)
        newlst = []
        for at in lst:
            try:
                res = chinese2arabic(at)
                newlst.append(res)
            except:
                pass
        lst = newlst
        if len(lst) > 0:
            text = lst[0]
            articles.append(text)
        else:
            articles.append("###")


    # articles = pd.Series(articles)
    # article_lst = articles.value_counts()[:150]
    # article_lst = article_lst.drop(["###", "零"])
