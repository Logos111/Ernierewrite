import json
import requests
import random
import hmac
import hashlib


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
    i = 0
    for sample in data:
        if i >= 10:
            break
        i = i + 1
        nowdata = {}
        history = sample["History"]
        question = sample["Question"]
        rewrite = sample["Rewrite"]
        history = "|".join(history)
        nowdata["history"] = get_translate(history)
        nowdata["question"] = get_translate(question)
        nowdata["rewrite"] = get_translate(rewrite)
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
    with open("result.json", "w") as file:
        json.dump(zhtraindata, file)
