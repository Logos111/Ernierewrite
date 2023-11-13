import re
import jsonlines
import time
import os
from get_seq_cls import ErnieForClsAndSeq
from paddlenlp.data import DataCollatorForTokenClassification
import paddle
from paddle.io import DataLoader, BatchSampler
import functools
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.datasets import MapDataset
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import ChunkEvaluator
import json

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as fp:
        label_vocab = json.load(fp)
    return label_vocab


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()

    for step, batch in enumerate(data_loader, start=1):
        input_ids, attention_mask, labels, classes, seq_len = batch['input_ids'], batch['attention_mask'], batch[
            'labels'], batch["classes"], batch["seq_len"]
        # 计算模型输出、损失函数值
        pooled_logits, sequence_logits = model(input_ids=input_ids,
                                               attention_mask=attention_mask)
        sequence_preds = paddle.argmax(sequence_logits, axis=-1)

        n_infer, n_label, n_correct = metric.compute(seq_len, sequence_preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())

    precision, recall, f1_score = metric.accumulate()
    print("sequence eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))

    model.train()
    return f1_score


def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx]['tokens'][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.startswith('B-'):
                    tags_out.append(t.split('-')[1])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


def char_to_token(text, tokens):
    """
    Get the index of the token in the encoded output comprising a character in the original text.
    :param text:
    :param tokens:
    :return: dict{char_idx: token_idx}
    """
    char_token_dict = {}
    token_idx = 1 if tokens[0] == '[CLS]' else 0
    token_text = tokens[token_idx]
    cur_pos = 0
    for char_idx in range(len(text)):
        if text[char_idx].lower() == token_text[cur_pos]:
            char_token_dict[char_idx] = token_idx
            # the end of current token
            if cur_pos == len(token_text) - 1:
                cur_pos = 0
                token_idx += 1
                token_text = tokens[token_idx]
                if token_text.startswith('##'):
                    token_text = token_text[2:]
            else:
                cur_pos += 1
        else:
            # the skipping char in the text must split tokens
            assert cur_pos == 0
    # All tokens have been attached with chars besides '[SEP]'
    assert token_idx == len(tokens) or tokens[token_idx] == '[SEP]'
    return char_token_dict


def char_idx_to_token_idx(char_token_dict, start, end):
    """
    Convert char indexing to token indexing.
    :param char_token_dict:
    :param start:
    :param end:
    :return:
    """
    token_ix_set = set()
    for char_ix in range(start, end):
        if char_token_dict.get(char_ix):
            token_ix = char_token_dict[char_ix]
            # white spaces have no token and will return None
            token_ix_set.add(token_ix)
            ixs = sorted(token_ix_set)
            # continuous indices
            assert len(ixs) == ixs[-1] - ixs[0] + 1
    return ixs


def align_annotations_with_bio(token_len, label_vocab, char_token_dict, annotations):
    """
    Align tokens' labeling to BIO scheme: B for beginning tokens,
        I for inside tokens. And convert char indexing to token indexing;
        note that a tokenizer may add special tokens, e.g., [CLS], [SEP], etc.
    :param token_len:
    :param char_token_dict:
    :param annotations:
    :return: tokens with aligned BIO tags.
    """
    # Make a list to store our labels the same length as our tokens
    aligned_labels = [label_vocab['O']] * token_len
    for anno in annotations:
        # A set that stores the token indices of the annotation
        token_list = char_idx_to_token_idx(char_token_dict, anno['start'], anno['end'])
        # the first token
        aligned_labels[token_list[0]] = label_vocab[f"B-{anno['tag']}"]
        # inside tokens
        for idx in range(1, len(token_list)):
            aligned_labels[token_list[idx]] = label_vocab[f"I-{anno['tag']}"]
    return aligned_labels


def get_samples(i, d, keep, q_type):
    text = d["input"]
    sample = {"NO": i, "text": text, "annotations": []}
    for k, v in d["output"].items():
        if k in keep:
            v = v if isinstance(v, str) else str(v)
            start_offset = text.find(v)
            if start_offset == -1:
                print(d, v)
            sample["annotations"].append({"tag": k, "start": start_offset, "end": start_offset + len(v)})
    sample["classes"] = q_type[d["output"]["domain"]]
    return sample


def read(data_path):
    ls = []
    with jsonlines.open(data_path, 'r') as fp:
        for obj in fp:
            ls.append(f"{obj}")
    print(f"原始长度：{len(ls)}")
    ls = [eval(l) for l in set(ls)]
    print(f"去重后长度：{len(ls)}")
    return ls


# 数据预处理函数-预测
def preprocess(example, tokenizer, max_seq_length=128):
    """
    Preprocess the original text with Ernie tokenizer
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    tokenized_input = tokenizer(example,
                                max_seq_len=max_seq_length,
                                return_attention_mask=True)
    return tokenized_input


# 数据预处理函数-预测
def preprocess_token(example, tokenizer, max_seq_length=128):
    """
    Preprocess the original text with Ernie tokenizer
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    tokenized_input = tokenizer(example,
                                max_seq_len=max_seq_length,
                                return_attention_mask=True)
    return tokenized_input


# 数据预处理函数-训练
def preprocess_function(example, tokenizer, label_vocab, max_seq_length=128):
    labels = example['labels']
    tokens = example['tokens']
    no_entity_id = label_vocab['O']
    tokenized_input = tokenizer(tokens, return_length=True, is_split_into_words=True,
                                max_seq_len=max_seq_length, return_attention_mask=True)
    # 保证label与input_ids长度一致
    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 != len(labels):
        print(len(tokenized_input['input_ids']), len(labels), tokens)
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['classes'] = example["classes"]
    # tokenized_input['seq_len'] = len(tokenized_input['labels'])

    return tokenized_input


# 数据预处理函数-训练
def preprocess_function_token(example, label_vocab, tokenizer, max_seq_length=128):
    """
    Preprocess the original text with Ernie tokenizer
    :param example:
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    tokenized_input = tokenizer(example['text'], return_length=True,
                                max_seq_len=max_seq_length,
                                return_attention_mask=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
    char_token_dict = char_to_token(example['text'], tokens)
    tokenized_input['labels'] = align_annotations_with_bio(len(tokens),
                                                           label_vocab,
                                                           char_token_dict,
                                                           example['annotations'])

    tokenized_input['classes'] = example["classes"]
    # tokenized_input['seq_len'] = len(tokenized_input['labels'])

    return tokenized_input


def is_english(c):
    pattern = re.compile('[a-zA-Z]')
    return pattern.match(c) is not None


def token_merge(token_text, token_lab, tmp):
    words = ""
    for j in range(len(token_text)):
        l = token_lab[j]
        t = token_text[j]
        if l.startswith("B-"):
            tmp_l = l.split("B-")[1]
            if j > 1 and token_lab[j - 1] != "O" and tmp_l == token_lab[j - 1].split("-")[1]:
                words += t.replace("##", "")
            else:
                words = t.replace("##", "")

            if j < len(token_text) - 1:
                if (token_lab[j + 1].startswith("B-")
                    and l != token_lab[j + 1]) or token_lab[j + 1].startswith("O"):
                    if not tmp.get(tmp_l):
                        tmp[tmp_l] = words
                    else:
                        tmp[tmp_l] = ",".join([words, tmp[tmp_l]])
                    words = ""
                # elif l == token_lab[j - 1]:
                #     words = token_text[j - 1] + words

        elif l.startswith("I-"):
            tmp_l = l.split("I-")[1]
            if is_english(token_text[j - 1]) and is_english(token_text[j]):
                words += " "

            words += t.replace("##", "")
            if j < len(token_text) - 1 and (token_lab[j + 1] == "O" or tmp_l != token_lab[j + 1].split("-")[1]):
                if not tmp.get(tmp_l):
                    tmp[tmp_l] = words
                else:
                    tmp[tmp_l] = ",".join([words, tmp[tmp_l]])
                words = ""
    return tmp


def char_merge(token_text, token_lab, tmp):
    words = ""
    for j in range(len(token_text)):
        l = token_lab[j]
        t = token_text[j]
        if l.startswith("B-"):
            words = t.replace("##", "")
            tmp_l = l.split("B-")[1]
            if j < len(token_text) - 1:
                if (token_lab[j + 1].startswith("B-")
                    and l != token_lab[j + 1]) or token_lab[j + 1].startswith("O"):
                    tmp[tmp_l] = words
                    words = ""
                elif l == token_lab[j - 1]:
                    words = token_text[j - 1] + words

        elif l.startswith("I-"):
            tmp_l = l.split("I-")[1]

            words += t.replace("##", "")
            if j < len(token_text) - 1 and not token_lab[j + 1].startswith("I-"):
                tmp[tmp_l] = words
                words = ""
    return tmp


class PredictorForChar:
    def __init__(self, model_name, model_state, vocab_path):
        self.label_vocab = load_vocab(vocab_path)
        self.vocab_label = {v: k for k, v in self.label_vocab.items()}
        self.label_list = list(self.label_vocab.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载最佳模型参数
        self.model = ErnieForClsAndSeq(model_name, len(self.label_list), 4)
        self.model.set_dict(paddle.load(model_state))

    def data_process(self, test_ds):
        trans_func = functools.partial(preprocess, tokenizer=self.tokenizer,
                                       label_vocab=self.label_vocab,
                                       max_seq_length=128)
        test_ds = test_ds.map(trans_func)

        # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
        collate_fn = DataCollatorForTokenClassification(tokenizer=self.tokenizer, label_pad_token_id=-100)

        # 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
        test_batch_sampler = BatchSampler(test_ds, batch_size=128, shuffle=False)
        test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
        return test_data_loader

    def get_data_loader(self, texts):
        ds = MapDataset(texts)
        data_loader = self.data_process(ds)
        return data_loader

    def predict(self, texts):
        data_loader = self.get_data_loader(texts)
        self.model.eval()
        res = []

        for step, batch in enumerate(data_loader, start=1):
            input_ids = batch['input_ids']
            # 计算模型输出、损失函数值
            pooled_logits, sequence_logits = self.model(input_ids=input_ids)
            sequence_preds = paddle.argmax(sequence_logits, axis=-1)
            pooled_preds = paddle.argmax(pooled_logits, axis=-1)

            for i in range(len(input_ids)):
                tmp = {}

                token_text = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                token_lab = [self.vocab_label[lab] for lab in sequence_preds[i].numpy()]
                tmp["token_text"] = token_text
                tmp["token_lab"] = token_lab
                tmp["classes"] = int(pooled_preds.numpy()[i])

                tmp = char_merge(token_text, token_lab, tmp)
                res.append(tmp)


class PredictorForToken:
    def __init__(self, model_name, model_state, vocab_path):
        self.label_vocab = load_vocab(vocab_path)
        self.vocab_label = {v: k for k, v in self.label_vocab.items()}
        self.label_list = list(self.label_vocab.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载最佳模型参数
        self.model = ErnieForClsAndSeq(model_name, len(self.label_list), 4)
        self.model.set_dict(paddle.load(model_state))

    def data_process(self, test_ds):
        trans_func = functools.partial(preprocess_token, tokenizer=self.tokenizer,
                                       max_seq_length=128)
        test_ds = test_ds.map(trans_func)

        # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
        collate_fn = DataCollatorForTokenClassification(tokenizer=self.tokenizer, label_pad_token_id=-100)

        # 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
        test_batch_sampler = BatchSampler(test_ds, batch_size=128, shuffle=False)
        test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
        return test_data_loader

    def get_data_loader(self, texts):
        test_ds = MapDataset(texts)
        test_data_loader = self.data_process(test_ds)
        return test_data_loader

    def predict(self, texts):
        data_loader = self.get_data_loader(texts)
        self.model.eval()
        res = []
        for step, batch in enumerate(data_loader, start=1):
            input_ids = batch['input_ids']
            # 计算模型输出、损失函数值
            pooled_logits, sequence_logits = self.model(input_ids=input_ids)
            sequence_preds = paddle.argmax(sequence_logits, axis=-1)
            pooled_preds = paddle.argmax(pooled_logits, axis=-1)

            for i in range(len(input_ids)):
                tmp = {}

                token_text = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                token_lab = [self.vocab_label[lab] for lab in sequence_preds[i].numpy()]
                tmp["token_text"] = token_text
                tmp["token_lab"] = token_lab

                tmp["classes"] = int(pooled_preds.numpy()[i])

                tmp = token_merge(token_text, token_lab, tmp)
                res.append(tmp)

        return res


class TrainerForChar:
    def __init__(self, model_name, vocab_path, batch_size=32, cls_w=1, seq_w=1,
                 learning_rate=2e-5, epochs=10, ckpt_dir="ernie_ckpt", max_seq_length=128):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cls_w = cls_w
        self.seq_w = seq_w
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.ckpt_dir = ckpt_dir
        self.label_vocab = load_vocab(vocab_path)
        self.vocab_label = {v: k for k, v in self.label_vocab.items()}
        self.label_list = list(self.label_vocab.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载最佳模型参数
        self.model = ErnieForClsAndSeq(model_name, len(self.label_list), 4)

    def data_process(self, ds):
        trans_func = functools.partial(preprocess_function, tokenizer=self.tokenizer,
                                       label_vocab=self.label_vocab,
                                       max_seq_length=self.max_seq_length)
        ds = ds.map(trans_func)

        # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
        collate_fn = DataCollatorForTokenClassification(tokenizer=self.tokenizer, label_pad_token_id=-100)

        # 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
        batch_sampler = BatchSampler(ds, batch_size=self.batch_size, shuffle=False)
        data_loader = DataLoader(dataset=ds, batch_sampler=batch_sampler, collate_fn=collate_fn)
        return data_loader

    def get_data_loader(self, train_path, test_path):
        train_ds = load_dataset(read, data_path=train_path, lazy=False)
        test_ds = load_dataset(read, data_path=test_path, lazy=False)

        train_data_loader = self.data_process(train_ds)
        test_data_loader = self.data_process(test_ds)
        return train_data_loader, test_data_loader

    def train(self, train_path, test_path):
        train_data_loader, test_data_loader = self.get_data_loader(train_path, test_path)
        # Adam优化器、交叉熵损失函数、ChunkEvaluator评价指标
        optimizer = paddle.optimizer.AdamW(learning_rate=self.learning_rate, parameters=self.model.parameters())
        criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=-100)
        seq_metric = ChunkEvaluator(label_list=self.label_list)
        # 开始训练
        best_f1_score = 0
        best_step = 0
        global_step = 0  # 迭代次数
        tic_train = time.time()
        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, attention_mask, labels, classes = batch['input_ids'], batch['attention_mask'], batch[
                    'labels'], batch["classes"]

                # 计算模型输出、损失函数值
                pooled_logits, sequence_logits = self.model(input_ids=input_ids,
                                                            attention_mask=attention_mask)
                pool_loss = paddle.mean(criterion(pooled_logits, classes))
                seq_loss = paddle.mean(criterion(sequence_logits, labels))
                loss = self.cls_w * pool_loss + seq_loss * self.seq_w

                # 每迭代10次，打印损失函数值、计算速度
                global_step += 1
                if global_step % 10 == 0:
                    with open("../data/train.log", "w", encoding="utf-8") as fp:
                        fp.write(f"分类loss：{pool_loss}, 序列loss:{seq_loss}")

                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
                    tic_train = time.time()

                # 反向梯度回传
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                # 每迭代200次，评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
                if global_step % 200 == 0:
                    save_dir = self.ckpt_dir
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    print('global_step', global_step, end=' ')
                    f1_score_eval = evaluate(self.model, seq_metric, test_data_loader)

                    if f1_score_eval > best_f1_score:
                        best_f1_score = f1_score_eval
                        best_step = global_step

                        # model.save_pretrained(save_dir)
                        paddle.save(self.model.state_dict(), f"{save_dir}/state.pdparams")
                        paddle.save(optimizer.state_dict(), f"{save_dir}/optimizer.pdopt")
                        self.tokenizer.save_pretrained(save_dir)

    def evaluate(self, test_path):
        test_ds = load_dataset(read, data_path=test_path, lazy=False)
        test_data_loader = self.data_process(test_ds)
        metric = ChunkEvaluator(label_list=self.label_list)
        pred_list = []
        clse = []
        for step, batch in enumerate(test_data_loader, start=1):
            input_ids, labels, lens, classes = batch['input_ids'], batch[
                'labels'], batch['seq_len'], batch['classes']

            pooled_logits, sequence_logits = self.model(input_ids=input_ids)
            sequence_preds = paddle.argmax(sequence_logits, axis=-1)
            pooled_preds = paddle.argmax(pooled_logits, axis=-1)

            n_infer, n_label, n_correct = metric.compute(lens, sequence_preds, labels)
            metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())

            pred_list.append(pooled_preds.numpy())
            clse.append(classes.numpy())

        precision, recall, f1_score = metric.accumulate()
        print("ERNIE 3.0 Medium 在msra_ner的test集表现 -precision: %f - recall: %f - f1: %f" % (precision, recall, f1_score))

        num = 0
        total = 0
        for x, y in zip(pred_list[0], clse[0]):
            if x == y:
                num += 1
            total += 1
        print(f"准确率：{num / total}")


class TrainerForToken:
    def __init__(self, model_name, vocab_path, batch_size=32, cls_w=1, seq_w=1,
                 learning_rate=2e-5, epochs=10, ckpt_dir="ernie_ckpt", max_seq_length=128):
        self.batch_size = batch_size
        self.cls_w = cls_w
        self.seq_w = seq_w
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.ckpt_dir = ckpt_dir
        self.label_vocab = load_vocab(vocab_path)
        self.vocab_label = {v: k for k, v in self.label_vocab.items()}
        self.label_list = list(self.label_vocab.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 加载最佳模型参数
        self.model = ErnieForClsAndSeq(model_name, len(self.label_list), 4)

    def data_process(self, ds):
        trans_func = functools.partial(preprocess_function_token, tokenizer=self.tokenizer,
                                       label_vocab=self.label_vocab,
                                       max_seq_length=self.max_seq_length)
        ds = ds.map(trans_func)

        # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
        collate_fn = DataCollatorForTokenClassification(tokenizer=self.tokenizer, label_pad_token_id=-100)

        # 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
        batch_sampler = BatchSampler(ds, batch_size=self.batch_size, shuffle=False)
        data_loader = DataLoader(dataset=ds, batch_sampler=batch_sampler, collate_fn=collate_fn)
        return data_loader

    def get_data_loader(self, train_path, test_path):
        train_ds = load_dataset(read, data_path=train_path, lazy=False)
        test_ds = load_dataset(read, data_path=test_path, lazy=False)

        train_data_loader = self.data_process(train_ds)
        test_data_loader = self.data_process(test_ds)
        return train_data_loader, test_data_loader

    def train(self, train_path, test_path):
        train_data_loader, test_data_loader = self.get_data_loader(train_path, test_path)
        # Adam优化器、交叉熵损失函数、ChunkEvaluator评价指标
        optimizer = paddle.optimizer.AdamW(learning_rate=self.learning_rate, parameters=self.model.parameters())
        criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=-100)
        seq_metric = ChunkEvaluator(label_list=self.label_list)
        # 开始训练
        best_f1_score = 0
        global_step = 0  # 迭代次数
        tic_train = time.time()
        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                
                input_ids, attention_mask, labels, classes = batch['input_ids'], batch['attention_mask'], batch[
                    'labels'], batch["classes"]

                # 计算模型输出、损失函数值
                pooled_logits, sequence_logits = self.model(input_ids=input_ids,
                                                            attention_mask=attention_mask)
                
                pool_loss = paddle.mean(criterion(pooled_logits, classes))
                seq_loss = paddle.mean(criterion(sequence_logits, labels))
                loss = self.cls_w * pool_loss + seq_loss * self.seq_w

                # 每迭代10次，打印损失函数值、计算速度
                global_step += 1
                if global_step % 10 == 0:
                    # with open("../data/train.log", "w", encoding="utf-8") as fp:
                    #     fp.write(f"分类loss：{pool_loss}, 序列loss:{seq_loss}")

                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
                    tic_train = time.time()

                # 反向梯度回传
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                # 每迭代200次，评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
                if global_step % 200 == 0:
                    save_dir = self.ckpt_dir
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    print('global_step', global_step, end=' ')
                    f1_score_eval = evaluate(self.model, seq_metric, test_data_loader)

                    if f1_score_eval > best_f1_score:
                        best_f1_score = f1_score_eval

                        # model.save_pretrained(save_dir)
                        paddle.save(self.model.state_dict(), f"{save_dir}/state.pdparams")
                        paddle.save(optimizer.state_dict(), f"{save_dir}/optimizer.pdopt")
                        self.tokenizer.save_pretrained(save_dir)

    def evaluate(self, test_path):
        test_ds = load_dataset(read, data_path=test_path, lazy=False)
        test_data_loader = self.data_process(test_ds)
        metric = ChunkEvaluator(label_list=self.label_list)
        pred_list = []
        clse = []
        for step, batch in enumerate(test_data_loader, start=1):
            input_ids, labels, lens, classes = batch['input_ids'], batch[
                'labels'], batch['seq_len'], batch['classes']

            pooled_logits, sequence_logits = self.model(input_ids=input_ids)
            sequence_preds = paddle.argmax(sequence_logits, axis=-1)
            pooled_preds = paddle.argmax(pooled_logits, axis=-1)

            n_infer, n_label, n_correct = metric.compute(lens, sequence_preds, labels)
            metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())

            pred_list.append(pooled_preds.numpy())
            clse.append(classes.numpy())

        precision, recall, f1_score = metric.accumulate()
        print("ERNIE 3.0 Medium 在msra_ner的test集表现 -precision: %f - recall: %f - f1: %f" % (precision, recall, f1_score))

        num = 0
        total = 0
        for x, y in zip(pred_list[0], clse[0]):
            if x == y:
                num += 1
            total += 1
        print(f"准确率：{num / total}")
        return precision, recall, f1_score, num / total


# dic = {
#     'token_text': ['[CLS]', 'ahc', '天', '猫', '官', '方', '旗', '舰', '店', 'b5', '品', '线', '5', '月', '的', '流', '量', '来', '源',
#                    '[SEP]'],
#     'token_lab': ['O', 'B-shop', 'I-shop', 'I-shop', 'I-shop', 'I-shop', 'I-shop', 'I-shop', 'I-shop', 'B-product_series',
#                   'I-product_series', 'I-product_series', 'B-origin_date', 'I-origin_date', 'O', 'B-index', 'I-index',
#                   'B-groupby_info', 'I-groupby_info', 'O'], 'classes': 2, 'shop': '线,ahc天猫官方旗舰店b5',
#     'product_series': '品', 'origin_date': '5月', 'index': '流量', 'groupby_info': '来源'}
# token_text = dic["token_text"]
# token_lab = dic["token_lab"]
# tmp = {}
# print(token_merge(token_text, token_lab, tmp))
