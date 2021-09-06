import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class SpamDataSet(Dataset):

    def __init__(self, file_path, tokenizer=None):
        self.file_path = file_path
        data = pd.read_csv(self.file_path)
        self.sentences = data['email'].tolist()
        self.labels = data['label'].tolist()
        if tokenizer is None:
            self.tokenizer = TfidfVectorizer(lowercase=True, stop_words=None)
            self.tokenized_sen = self.tokenizer.fit_transform(self.sentences)
        else:
            self.tokenizer = tokenizer
            self.tokenized_sen = self.tokenizer.transform(self.sentences)
        self.vocabulary_size = len(self.tokenizer.vocabulary_)

    def __getitem__(self, item):
        cur_sen = self.tokenized_sen[item]
        cur_sen = torch.FloatTensor(cur_sen.toarray()).squeeze()
        label = self.labels[item]
        # label = torch.Tensor(label)
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.sentences)
