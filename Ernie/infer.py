from fastapi import FastAPI
from typing import Optional
import os

from pydantic import BaseModel
import uvicorn

from utils import PredictorForToken
import psutil
import setproctitle

# 获取当前进程ID
pid = str(os.getpid())
# 获取进程对象
p = psutil.Process(int(pid))
# 设置进程名字
setproctitle.setproctitle("PredictorForToken")


# model_name = "ernie-3.0-medium-zh"
model_name = "/data/jupyter/LLM/models/ernie-3.0-medium-zh"
model_state = '/data/jupyter/LLM/models/ernie_ckpt/state.pdparams'
vocab_path = "/data/jupyter/LLM/models/ernie_ckpt/label_vocab.json"


class Parse(BaseModel):
    texts: list


app = FastAPI()
predictor = PredictorForToken(model_name, model_state, vocab_path)

q_type = {"商品": 0, "活动": 1, "内容": 2, "客诉": 3}
invert_type ={v: k for k, v in q_type.items()}

@app.post("/get_text_parse")
async def parse_text(parse: Parse):
    texts = parse.texts
    res = []
    for tmp in predictor.predict(texts):
        tmp["classes"] = invert_type[tmp["classes"]]
        res.append(tmp)
    return {"result_list": res}


if __name__ == '__main__':
    uvicorn.run("infer:app", host='127.0.0.1', port=5050, reload=False)
