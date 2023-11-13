import random
from tqdm import tqdm
import jsonlines
import json
import pandas as pd
from get_sample import *

def _init_mysql_connect():
    import pymysql
    # 打开数据库连接
    conn = pymysql.connect(host='localhost', port=3306, user='root', 
                           password='K8*Vi!6t&M^1L', db='watsons')
    return conn

with open("./slot.json", "r") as fp:
    slot_dic = json.load(fp)

with open("./slot_col_dic.json", "r") as fp:
    slot_col_dic = json.load(fp)

conn = _init_mysql_connect()
sql = "select * from ITF_WDC2LLMFM_PRODUCT"
df = pd.read_sql(sql=sql, con=conn)

slot_ele_dic = {}
for k, v in slot_dic.items():
    if slot_col_dic.get(k):
        k_ = slot_col_dic[k].upper()
    else:
        k_ = k.upper()

    if k_ in df.columns:
        slot_ele_dic[k.lower()] = list(df[k_].dropna().unique())

slot_ele_dic["skin_type"] = ["油皮", "干皮", "中性"]
slot_ele_dic["shop"] = ["深圳门店"]
slot_ele_dic["index"] = ["销量"]
slot_ele_dic["num"] = list(range(1, 11)) + ["一", "二", "三", "四", "五"]
slot_ele_dic["date"] = [f"{i}月" for i in range(1, 13)]
slot_ele_dic["activity"] = ["母亲节","电商日", "双十一", "618", "开学季"]
slot_ele_dic["customer"] = ["会员", "新人", "老客", "老用户"]
slot_ele_dic["season"] = ["夏季","春季","秋季","冬季", "秋冬", "春夏"]
slot_ele_dic["star"] = ["刘德华","陈乔恩","张艺兴","陈伟霆", "赵丽颖", "郑爽", "王一博"]
slot_ele_dic["age"] = ["20-30岁","中年","青少年", "40岁以上"]
slot_ele_dic["pain"] = ["脱发","脱皮","唇干燥","长痘","发黑","皱纹","起皱"]
slot_ele_dic["city"] = ["深圳","广州","上海", "北京"]
slot_ele_dic["complaint_question"] = ["包装破损","质量不行", "发货太慢", "物流太慢", "刚买就降价", "商品与描述不符合"]
slot_ele_dic["complaint_plan"] = ["退货退款", "换货", "退款", "退货", "维修", "延保", "补偿", "技术支持"]
slot_ele_dic["lower"] = ["以内", "小于等于", "不超过", "最多", "不大于", "以下"]
slot_ele_dic["upper"] = ["至少", "大于等于", "不少于", "最小", "以上"]

def save_label_vocab(path, slot_dic):
    bio = []
    for k, _ in slot_dic.items():
            b = f"B-{k}"
            i = f"I-{k}"
            bio.append(b)
            bio.append(i)
    bio_labels = {k: i for i, k in enumerate(bio)}
    bio_labels["O"] = len(bio)

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(bio_labels, fp, ensure_ascii=False)

save_label_vocab(path="./data/label_vocab.json", slot_dic=slot_dic)

with open("./data/label_vocab.json", "r", encoding="utf-8") as fp:
    bio_labels = json.load(fp)

def get_samples(d, keep, q_type):
    text = d["text"]
    sample = {"text": text, "annotations": []}
    for k, v in d["output"].items():
            if k in keep:
                v = v if isinstance(v, str) else str(v)
                for v_ in v.split(","):
                    start_offset = text.find(v_)
                    if start_offset == -1:
                        print(d, v_)
                    if len(v_):
                        sample["annotations"].append({"tag": k, "start": start_offset, 
                                                    "end": start_offset + len(v_)})
    sample["classes"] = q_type[d["output"]["intent_1"]]
    return sample


q_type = {"商品": 0, "活动": 1, "内容": 2, "客诉": 3}
samples = []
ds = []

for i in tqdm(range(1000)):
    ds = [get_activity(**slot_ele_dic), get_knowledge(**slot_ele_dic), 
          get_product(**slot_ele_dic), get_servicer(**slot_ele_dic)]
    for data in ds:
        for d in data:
            sample = get_samples(d, keep=list(slot_dic.keys()), q_type=q_type)
            samples.append(sample)

samples = list([eval(s) for s in set([f"{l}" for l in samples])])
random.shuffle(samples)
num = int(len(samples) * 0.8)

with jsonlines.open(f"./data/chat-ext-train.jsonl", "w") as fp:
    for s in samples[:num]:
        fp.write(s)

with jsonlines.open(f"./data/chat-ext-dev.jsonl", "w") as fp:
    for s in samples[num:]:
        fp.write(s)

# print(samples)
