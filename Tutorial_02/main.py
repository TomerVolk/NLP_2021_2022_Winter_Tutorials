from dataset import SpamDataSet
from train import train
from torch.optim import Adam
from model import SpamClassifier


if __name__ == '__main__':
    train_ds = SpamDataSet('data/train_spam.csv')
    test_ds = SpamDataSet('data/test_spam.csv', tokenizer=train_ds.tokenizer)
    datasets = {"train": train_ds, "test": test_ds}
    model = SpamClassifier(num_classes=2, vocab_size=train_ds.vocabulary_size)
    optimizer = Adam(params=model.parameters())
    train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=15)
