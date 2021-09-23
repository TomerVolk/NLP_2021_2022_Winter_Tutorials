from typing import List
from collections import Counter
from pprint import pprint
import os
import pandas as pd


def split_to_reports(data_str: List[str]):
    reports = []
    cur_report = {}

    # topic_tags = {"TOPICS": 'topics', "PLACES": 'places', "PEOPLE": 'people', "ORGS": 'organizations',
    #               "EXCHANGES": 'exchanges', "COMPANIES": 'companies'}
    topic_tags = {"TOPICS": 'topics'}
    cur_body = []
    for row in data_str:
        row = row.strip()
        if row.startswith("<REUTERS"):
            row = row.split('CGISPLIT')[1]
            train_test = row.split('"')[1].split('"')[0]
            cur_report["train test"] = train_test
        if row.startswith("</REUTERS"):
            reports.append(cur_report)
            cur_report = {}
        for topic, name in topic_tags.items():
            if row.startswith(f"<{topic}"):
                row = row.strip().replace(f"<{topic}>", '').replace(f"</{topic}>", '')
                if len(row) == 0:
                    cur_report[name] = None
                    break
                cur_topics = [x.split("</D>")[0] for x in row.split("<D>")[1:]]
                cur_report[name] = cur_topics
                break
        if "<BODY>" in row and "</BODY>" in row:
            cur_txt = row.split("<BODY>")[1].split("</BODY>")[0]
            cur_report['text'] = cur_txt
            cur_body = []
        elif "<BODY>" in row:
            cur_txt = row.split("<BODY>")[1]
            cur_body.append(cur_txt)
        elif "</BODY>" in row:
            cur_txt = row.split("</BODY>")[0]
            cur_body.append(cur_txt)
            cur_body = " ".join(cur_body)
            cur_report['text'] = cur_body
            cur_body = []
        elif len(cur_body) > 0:
            cur_body.append(row)
    return reports


def organize_reuters_file(data_path):
    with open(data_path, "r", encoding='utf8', errors='ignore') as f:
        data_str = f.readlines()
    data_str = data_str[1:]
    reports = split_to_reports(data_str)
    reports = pd.DataFrame(reports)
    reports.dropna(axis=0, inplace=True)
    return reports


def organize_reuters():
    data = []
    reuters_path = "data"
    for file in os.listdir(reuters_path):
        if "reut2" not in file or '.sgm' not in file:
            continue
        file_path = f"{reuters_path}/{file}"
        cur_data = organize_reuters_file(file_path)
        data.append(cur_data)
    data = pd.concat(data).reset_index(drop=True)
    data['is train'] = data['train test'] == 'TRAINING-SET'
    train, test = data[data['is train']], data[~data['is train']]
    train, test = train.copy(), test.copy()
    train.drop(columns='is train', inplace=True)
    test.drop(columns='is train', inplace=True)
    # topics = train['topics'].tolist()
    # topics = sum(topics, [])
    # topic_count = Counter(topics)
    # pprint(topic_count)
    train.to_csv(f"{reuters_path}/train.csv", index_label=False, index=False)
    test.to_csv(f"{reuters_path}/test.csv", index_label=False, index=False)


if __name__ == '__main__':
    organize_reuters()