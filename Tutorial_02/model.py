from torch import nn


class SpamClassifier(nn.Module):

    def __init__(self, vocab_size, padding_idx, num_classes, emb_dim=100):
        super(SpamClassifier, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.first_layer = nn.Linear(emb_dim, 200)
        self.second_layer = nn.Linear(200, num_classes)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        emb = self.emb(input_ids)
        emb = emb.mean(dim=1)
        x = self.first_layer(emb)
        x = self.activation(x)
        x = self.second_layer(x)
        loss = self.loss(x, labels)
        return x, loss
