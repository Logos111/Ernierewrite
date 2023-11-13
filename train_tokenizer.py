import os
import pandas as pd
from utils import TrainerForToken

import psutil
import setproctitle

# 获取当前进程ID
pid = str(os.getpid())
# 获取进程对象
p = psutil.Process(int(pid))
# 设置进程名字
setproctitle.setproctitle("TrainErnie")


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_path = "./CANARD_Release/chat-ext-train.jsonl"
test_path = "./CANARD_Release/chat-ext-dev.jsonl"
vocab_path = "./data/label_vocab.json"
# model_name = "ernie-3.0-xbase-zh"
model_name = "ernie-3.0-medium-zh"

learning_rates = [2e-4, 2e-5, 2e-6]
batch_sizes = [32]
seq_ws = [1.0]
cls_ws = [1.0]

# 保存最佳参数组合到.csv文件
output_path = "./data/best_params.csv"
dic = {
    "learning_rate": [],
    "batch_size": [],
    "seq_w": [],
    "cls_w": [],
    "P": [],
    "R": [],
    "F1": [],
    "ACC": [],
}
# 嵌套循环遍历所有可能的参数组合
for lr in learning_rates:
    for bs in batch_sizes:
        for seq_w in seq_ws:
            for cls_w in cls_ws:
                # try:
                # 使用当前参数组合进行训练
                trainer = TrainerForToken(
                    model_name,
                    vocab_path,
                    batch_size=bs,
                    seq_w=seq_w,
                    cls_w=cls_w,
                    learning_rate=lr,
                    epochs=10,
                    ckpt_dir="ernie_ckpt",
                    max_seq_length=1024,
                )

                trainer.train(train_path, test_path)

                # 在测试集上进行评估
                precision, recall, f1_score, acc = trainer.evaluate(test_path)

                dic["learning_rate"].append(lr)
                dic["batch_size"].append(bs)
                dic["seq_w"].append(seq_w)
                dic["cls_w"].append(cls_w)
                dic["P"].append(precision)
                dic["R"].append(recall)
                dic["F1"].append(f1_score)
                dic["ACC"].append(acc)
                print(
                    f"lr:{lr}, batch_size:{bs}, seq_w:{seq_w}, cls_w:{cls_w} -precision: {precision} - recall: {recall} - "
                    f"f1: {f1_score},acc： {acc}"
                )
                # 跑一次记录一次，以防跑模型中断造成记录丢失
                df = pd.DataFrame(dic)
                df.to_csv(output_path, index=False)
            # except Exception as e:
            #     print(e)
