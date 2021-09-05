import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict


class SpamDataSet(Dataset):

    def __init__(self, file_path, vocabulary=None, vocabulary_threshold=5, max_seq_len=None):
        self.file_path = file_path
        self.vocabulary = vocabulary
        self.vocab_thresh = vocabulary_threshold
        data = pd.read_csv(self.file_path)
        self.sentences = data['email'].tolist()
        self.labels = data['label'].tolist()
        self.tokenized_sen = []
        self.unk_token = '<UNK>'
        self.padding_token = '<PAD>'
        self.tokenize()
        self.max_len = max_seq_len
        if max_seq_len is None:
            self.max_len = max([len(x) for x in self.tokenized_sen])
        pass

    def tokenize(self):
        self.get_vocab()
        for sen in self.sentences:
            tokenized = []
            for word in sen.split():
                if word not in self.vocabulary:
                    word = self.unk_token
                tokenized.append(self.vocabulary[word])
            self.tokenized_sen.append(tokenized)

    def get_vocab(self):
        if self.vocabulary is not None:
            return
        word_count = defaultdict(int)
        for sen in self.sentences:
            sen: str
            sen = sen.lower()
            for word in sen.split():
                word_count[word] += 1
        self.vocabulary = {self.unk_token: 0, self.padding_token: 1}

        counter = 2
        for word, val in word_count.items():
            if val > self.vocab_thresh:
                self.vocabulary[word] = counter
                counter += 1
        counter += 1

    def __getitem__(self, item):
        cur_sen = self.tokenized_sen[item]
        padding_len = self.max_len - len(cur_sen)
        cur_sen = cur_sen + [self.vocabulary[self.padding_token]] * padding_len
        cur_sen = torch.LongTensor(cur_sen)
        label = self.labels[item]
        # label = torch.Tensor(label)
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.sentences)
