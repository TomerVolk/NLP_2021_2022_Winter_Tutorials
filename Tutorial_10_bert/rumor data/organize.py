import os
import pickle
from os.path import isdir
from collections import defaultdict

import pandas as pd


def organize():
    text, labels = defaultdict(list), defaultdict(list)
    for folder in os.listdir('.'):
        if not isdir(folder):
            continue
        for file in os.listdir(folder):
            if file == "desktop.ini":
                continue
            with open(f"{folder}/{file}", "rb") as f:
                cur_data = pickle.load(f)
            cur_txt, cur_labels = cur_data
            text[file] += cur_txt
            labels[file] += cur_labels
    for key in text:
        data = pd.DataFrame.from_dict({"text": text[key], 'labels': labels[key]})
        data.to_csv(f'{key}.csv', index_label=False, index=False)


if __name__ == '__main__':
    organize()
