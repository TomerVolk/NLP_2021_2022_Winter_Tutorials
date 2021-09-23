import pandas as pd
import gzip
from sklearn.model_selection import train_test_split

DATA_PATH = 'reviews_Amazon_Instant_Video_5.json.gz'


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def create_csv():
    data = get_df(DATA_PATH)
    data = data[['reviewText', 'overall']]
    data['label'] = data['overall'] > 3
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    data.drop(columns='overall', inplace=True)
    data.replace({False: "Negative", True: "Positive"}, inplace=True)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv('train.csv', index_label=False, index=False)
    test.to_csv('test.csv', index_label=False, index=False)


if __name__ == '__main__':
    create_csv()
