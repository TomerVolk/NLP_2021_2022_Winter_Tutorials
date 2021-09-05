import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()

                cur_num_correct = accuracy_score(batch['labels'].cpu().view(-1), pred.view(-1), normalize=False)

                running_loss += loss.item() * batch_size
                running_acc += cur_num_correct

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = running_acc / len(data_sets[phase])

            epoch_acc = round(epoch_acc, 5)
            if phase.title() == "test":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
        print()

    print(f'Best Validation Accuracy: {best_acc:4f}')

    return model
