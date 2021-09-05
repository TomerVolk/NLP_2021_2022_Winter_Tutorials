from dataset import SpamDataSet
from train import train
from torch.optim import Adam
from model import SpamClassifier


if __name__ == '__main__':
    train_ds = SpamDataSet('data/train_spam.csv', vocabulary=None, vocabulary_threshold=5)
    test_ds = SpamDataSet('data/test_spam.csv', vocabulary=train_ds.vocabulary)
    datasets = {"train": train_ds, "test": test_ds}
    model = SpamClassifier(padding_idx=train_ds.vocabulary[train_ds.padding_token], num_classes=2,
                           vocab_size=len(train_ds.vocabulary))
    optimizer = Adam(params=model.parameters())
    train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=15)
