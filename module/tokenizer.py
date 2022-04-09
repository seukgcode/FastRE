from typing import List, Tuple
from tqdm import tqdm
import numpy as np


class Tokenizer(object):
    def __init__(self, glove_path: str, glove_dim: int):
        print("Tokenizer initialization......")
        self.glove_path = glove_path
        self.glove_dim = glove_dim
        self.id2word = {0: "<PAD>", 1: "<UNK>"}
        self.word2vec, self.id2vec = {}, {}
        with open(self.glove_path, encoding='utf-8') as gf:
            for idx, glove in tqdm(enumerate(gf)):
                word, vec = glove.split(maxsplit=1)
                self.id2word[idx + 2] = word
                vec = [float(s) for s in vec.split(' ')]
                self.word2vec[word] = vec
        self.word2vec["<PAD>"] = [0.0] * self.glove_dim
        self.word2vec["<UNK>"] = [0.0] * self.glove_dim
        self.word2id = {value: int(key) for key, value in self.id2word.items()}
        for idx in range(len(self.id2word)):
            self.id2vec[idx] = self.word2vec.get(self.id2word[idx], self.word2vec.get("<UNK>"))
        self.id2vec = np.array([self.id2vec[index] for index in range(len(self.id2word))])
        print("Tokenizer is done.")

    def tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        tokens = text.lower().strip().split()
        token_ids = [self.word2id.get(word, 1) for word in tokens]
        return tokens, token_ids

    def seq2vec(self, batch_token_ids: np.ndarray) -> np.ndarray:
        return self.id2vec[batch_token_ids]

