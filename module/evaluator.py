import os
import json
import codecs
import time
from typing import Tuple
from tqdm import tqdm
import numpy as np
from paddle import fluid
from .dataset import Dataset
from .model import SModel, OPModel
from .utils import partial


class Evaluator(object):
    def __init__(self, dataset: Dataset, name: str):
        self.dataset = dataset
        self.name = name
        self.best = 0.0

    def evaluate(self, path: str, s_model: SModel, op_model: OPModel, p: str) -> Tuple[float, float, float, float, float]:
        orders = ["subject", "predicate", "object"]
        a, b, c = 1e-10, 1e-10, 1e-10
        all_res = []
        total_time = 0.0
        with codecs.open(path, 'r', encoding='utf-8') as f:
            all_data = json.loads(f.read())
            for sample in tqdm(iter(all_data)):
                text = sample["text"]
                each_res = []
                s = time.time()
                tmp = self.extract_items(text, s_model, op_model)
                total_time = total_time + time.time() - s
                each_res.extend(tmp)
                r = set(each_res)
                t = set([(e["subject"], e["predicate"], e["object"]) for e in sample["spo_list"]])
                if p == "PM":
                    r, t = partial(r, t)
                a += len(r & t)
                b += len(r)
                c += len(t)
                all_res.append({
                    'text': sample['text'],
                    'spo_list': [
                        dict(zip(orders, spo)) for spo in t
                    ],
                    'spo_list_pred': [
                        dict(zip(orders, spo)) for spo in r
                    ],
                    'new': [
                        dict(zip(orders, spo)) for spo in r - t
                    ],
                    'lack': [
                        dict(zip(orders, spo)) for spo in t - r
                    ]
                })
        with codecs.open("./predict.json", 'w', encoding='utf-8') as f:
            json.dump(all_res, f, ensure_ascii=False, indent=4)

        if self.best < round(2 * a / (b + c), 4):
            self.best = round(2 * a / (b + c), 4)
            save_path = os.path.join(os.getcwd(), f"{path.split('/')[2]}_best_model")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sub_save_path = os.path.join(save_path, "SModel")
            fluid.save_dygraph(s_model.state_dict(), sub_save_path)
            obj_save_path = os.path.join(save_path, "OPModel")
            fluid.save_dygraph(op_model.state_dict(), obj_save_path)
        return round(2 * a / (b + c), 4), round(a / b, 4), round(a / c, 4), self.best, total_time

    def extract_items(self, text, s_model, op_model):
        _, token_ids = self.dataset.tokenizer.tokenize(text)
        token_ids = token_ids[:self.dataset.max_len]
        text = text.split()
        token_len = len(token_ids)
        token_ids = np.array([token_ids])
        sent_vec = fluid.dygraph.to_variable(self.dataset.tokenizer.seq2vec(token_ids)).astype("float32")
        token_ids = fluid.dygraph.to_variable(token_ids).astype("int64")
        _sub_head, _sub_tail, features, mask = s_model(token_ids, sent_vec)
        _th_head, _th_tail = _sub_head.numpy()[0, :token_len, :1], _sub_tail.numpy()[0, :token_len, :1]
        _sub_head, _sub_tail = _sub_head.numpy()[0, :token_len, 1:], _sub_tail.numpy()[0, :token_len, 1:]
        _subjects = []
        for _sub_id in range(self.dataset.num_subs):
            _sub_head_cur = _sub_head[:, _sub_id]
            _sub_tail_cur = _sub_tail[:, _sub_id]
            _th_head_cur = _th_head[:, 0]
            _th_tail_cur = _th_tail[:, 0]
            _sub_head_cur = np.where(_sub_head_cur > _th_head_cur)[0]
            _sub_tail_cur = np.where(_sub_tail_cur > _th_tail_cur)[0]
            for index, start in enumerate(_sub_head_cur):
                end_list = _sub_tail_cur[_sub_tail_cur >= start]
                if len(end_list) > 0:
                    if index + 1 < len(_sub_head_cur):
                        max_index = _sub_head_cur[index + 1]
                    else:
                        max_index = _sub_tail_cur[-1] + 2
                    for end in end_list:
                        if end < max_index and end - start < 20:
                            _subjects.append((start, end, _sub_id))

        res = []
        if _subjects:
            features_batch = fluid.layers.expand(features, [len(_subjects), 1, 1])
            mask = fluid.layers.expand(mask, [len(_subjects), 1, 1])
            _k1, _k2, _sub_id = np.array(_subjects).T.reshape((3, -1, 1))
            _k1_tensor = fluid.dygraph.to_variable(_k1).astype("int64")
            _k2_tensor = fluid.dygraph.to_variable(_k2).astype("int64")
            _sub_id_tensor = fluid.dygraph.to_variable(_sub_id).astype("int64")
            _o1, _o2 = op_model(features_batch, _sub_id_tensor, _k1_tensor, _k2_tensor, mask)
            _th_head, _th_tail = _o1.numpy()[:, :token_len, :1], _o2.numpy()[:, :token_len, :1]
            _o1, _o2 = _o1.numpy()[:, :token_len, 1:], _o2.numpy()[:, :token_len, 1:]
            for i, _subject in enumerate(_subjects):
                _oo1, _oo2 = np.where(_o1[i] > _th_head[i]), np.where(_o2[i] > _th_tail[i])
                for _ooo1, _c1 in zip(*_oo1):
                    for _ooo2, _c2 in zip(*_oo2):
                        if _ooo1 <= _ooo2 and _c1 == _c2:
                            if _ooo2 - _ooo1 > 20 or _ooo2 == token_len - 1:
                                continue
                            res.append((" ".join(text[_subject[0]:_subject[1] + 1]), self.dataset.id2rel[_c2],
                                        " ".join(text[_ooo1: _ooo2 + 1])))
                            break

            return res
        else:
            return []
