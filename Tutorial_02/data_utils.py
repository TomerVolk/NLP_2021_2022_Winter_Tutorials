import pandas as pd
from sklearn.model_selection import train_test_split


def balance_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna().drop_duplicates()
    positive = data[data['label'] == 1]
    negative = data[data['label'] == 0]
    negative = negative.sample(n=len(positive), random_state=42)
    data = positive.append(negative)
    train, test = train_test_split(data, random_state=42, test_size=0.2)
    train.to_csv('data/train_spam.csv')
    test.to_csv('data/test_spam.csv')
    pass


if __name__ == '__main__':
    balance_data('data/spam_or_not_spam.csv')
