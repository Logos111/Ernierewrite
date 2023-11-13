import json
import requests
import random
import hmac
import hashlib
from tqdm import tqdm


def mdsign(data):
    md = hashlib.md5()
    md.update(data.encode("utf-8"))
    return md.hexdigest()


def get_translate(query):
    APPID = "20210926000956961"
    key = "usw3vYr5INJhlUjpO4jn"
    salt = str(random.randint(10000, 99999))
    sign = mdsign(APPID + query + salt + key)
    payload = {
        "appid": APPID,
        "q": query,
        "from": "en",
        "to": "zh",
        "salt": salt,
        "sign": sign,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(
        "https://fanyi-api.baidu.com/api/trans/vip/translate",
        params=payload,
        headers=headers,
    )
    result = r.json()
    return result


def get_zhdataset(data):
    zhdataset = []
    for sample in tqdm(data[:10], desc="Processing"):
        nowdata = {}
        history = sample["History"]
        question = sample["Question"]
        rewrite = sample["Rewrite"]
        history = "|".join(history)
        nowdata["history"] = get_translate(history)["trans_result"][0]["dst"]
        nowdata["question"] = get_translate(question)["trans_result"][0]["dst"]
        nowdata["rewrite"] = get_translate(rewrite)["trans_result"][0]["dst"]

    zhdataset.append(nowdata)
    return zhdataset


if __name__ == "__main__":
    file = "./canard/data/release/"
    trainpath = "train.json"
    devpath = "dev.json"
    testpath = "test.json"
    outputpath = "train_zh.json"

    with open(file + trainpath, "r") as f:
        traindata = json.load(f)
    with open(file + devpath, "r") as f:
        devdata = json.load(f)
    with open(file + testpath, "r") as f:
        testdata = json.load(f)

    zhtraindata = get_zhdataset(traindata)
    with open("result.json", "a+", encoding="utf-8") as file:
        json.dump(zhtraindata, file, indent=4, ensure_ascii=False)
