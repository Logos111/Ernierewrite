import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import TrainerForToken

test_path = "../data/chat-ext-test.jsonl"

vocab_path = "../data/label_vocab.json"
# model_name = "ernie-3.0-xbase-zh"
model_name = "ernie-3.0-medium-zh"
model_state = 'ernie_ckpt/model_state.pdparams'

predictor = TrainerForToken(model_name, model_state, vocab_path)
predictor.evaluate(test_path)
