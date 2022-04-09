import json
import codecs
from random import choice
import numpy as np

from .tokenizer import Tokenizer
from .utils import find, pad


class Dataset(object):
    def __init__(self, path: str, schemas_path: str, tokenizer: Tokenizer, max_len: int, batch_size: int, name: str):
        print("Dataset initialization......")
        self.data = json.load(codecs.open(path, "r", encoding="utf-8"))
        self.schemas = json.load(codecs.open(schemas_path, "r", encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.name = name
        self.id2rel = {int(i): j for i, j in self.schemas["id2relation"].items()}
        self.rel2id = self.schemas["relation2id"]
        self.relation_set = set(self.rel2id.keys())
        self.num_relations = len(self.id2rel)
        self.id2subject = {int(i): j for i, j in self.schemas["id2subject"].items()}
        self.subject2id = self.schemas["subject2id"]
        self.num_subs = len(self.id2subject)
        self.id2object = {int(i): j for i, j in self.schemas["id2object"].items()}
        self.object2id = self.schemas["object2id"]
        self.num_objs = len(self.id2object)
        self.rel2so = self.schemas["rel2so"]
        self.predicates = {}
        for d in self.data:
            for spo in d['spo_list']:
                if spo["predicate"] not in self.predicates:
                    self.predicates[spo["predicate"]] = []
                self.predicates[spo["predicate"]].append(spo)
        print("Dataset is done.")

    def random_generate(self, d):
        r = np.random.random()
        if r > 0.5:
            return d
        else:
            k = np.random.randint(len(d["spo_list"]))
            spi = d["spo_list"][k]
            k = np.random.randint(len(self.predicates[spi["predicate"]]))
            spo = self.predicates[spi["predicate"]][k]
            F = lambda s: s.replace(spi["subject"], spo["subject"]).replace(spi["object"], spo["object"])
            text = F(d['text'])
            spo_list = [{"subject": F(sp["subject"]), "predicate": sp["predicate"], "object": F(sp["object"])} for sp in
                        d["spo_list"]]
            return {'text': text, "spo_list": spo_list}

    def generate_batch(self):
        def __reader__():
            indexes = list(range(len(self.data)))
            np.random.shuffle(indexes)
            token_ids_batch, sub_heads_batch, sub_tails_batch = [], [], []
            sub_id_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], []
            for idx in indexes:
                if self.name in ["NYT11"]:
                    line = self.random_generate(self.data[idx])
                else:
                    line = self.data[idx]
                tokens, token_ids = self.tokenizer.tokenize(line["text"])
                tokens, token_ids = tokens[:self.max_len], token_ids[:self.max_len]
                s2po_map = {}
                for spo in line['spo_list']:
                    p = spo["predicate"]
                    if p not in self.relation_set:
                        continue
                    sub, _ = self.tokenizer.tokenize(spo["subject"])
                    obj, _ = self.tokenizer.tokenize(spo["object"])
                    sub_head_idx, sub_tail_idx = find(tokens, sub)
                    obj_head_idx, obj_tail_idx = find(tokens, obj)
                    s_id = self.subject2id[self.rel2so[p]["subject"]]
                    o_id = self.object2id[self.rel2so[p]["object"]]
                    p = self.rel2id[p]
                    if sub_head_idx != -1 and obj_head_idx != -1:
                        sub_key = (sub_head_idx, sub_tail_idx - 1, s_id)
                        if sub_key not in s2po_map:
                            s2po_map[sub_key] = []
                        s2po_map[sub_key].append((obj_head_idx, obj_tail_idx - 1, p, s_id, o_id))
                if s2po_map:
                    sub_heads = np.zeros((len(token_ids), self.num_subs))
                    sub_tails = np.zeros((len(token_ids), self.num_subs))
                    for s_key, op in s2po_map.items():
                        for each in op:
                            sub_id = each[3]
                            sub_heads[s_key[0]][sub_id] = 1
                            sub_tails[s_key[1]][sub_id] = 1
                    sub_head, sub_tail, sub_id = choice(list(s2po_map.keys()))
                    obj_heads = np.zeros((len(token_ids), self.num_relations))
                    obj_tails = np.zeros((len(token_ids), self.num_relations))
                    for po in s2po_map.get((sub_head, sub_tail, sub_id), []):
                        obj_heads[po[0]][po[2]] = 1
                        obj_tails[po[1]][po[2]] = 1
                    token_ids_batch.append(token_ids)
                    sub_heads_batch.append(sub_heads)
                    sub_tails_batch.append(sub_tails)
                    sub_id_batch.append([sub_id])
                    sub_head_batch.append([sub_head])
                    sub_tail_batch.append([sub_tail])
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)

                    if len(token_ids_batch) == self.batch_size or idx == indexes[-1]:
                        token_ids_batch = pad(token_ids_batch, 0, "int64")
                        sent_vec_batch = self.tokenizer.seq2vec(token_ids_batch)
                        sub_heads_batch = pad(sub_heads_batch, np.zeros(self.num_subs), "float32")
                        sub_tails_batch = pad(sub_tails_batch, np.zeros(self.num_subs), "float32")
                        sub_id_batch = np.array(sub_id_batch)
                        sub_head_batch = np.array(sub_head_batch)
                        sub_tail_batch = np.array(sub_tail_batch)
                        obj_heads_batch = pad(obj_heads_batch, np.zeros(self.num_relations), "float32")
                        obj_tails_batch = pad(obj_tails_batch, np.zeros(self.num_relations), "float32")
                        yield \
                            token_ids_batch, sent_vec_batch, sub_heads_batch, sub_tails_batch, \
                            sub_id_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch
                        token_ids_batch, sub_heads_batch, sub_tails_batch = [], [], []
                        sub_id_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], []
        return __reader__
