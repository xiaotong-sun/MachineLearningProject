import torch
from matplotlib import pyplot as plt
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn
import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0
    i = 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            i += 1
            predicted_label = model(text, offsets)
            global loss_temp
            total_loss += (criterion(predicted_label, label)).sum().item()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        loss_temp.append(total_loss / i)
    return total_acc / total_count


def evaluate_test(dataloader):
    model.eval()
    total_count = 0
    acc = [0, 0, 0, 0]

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            for j in range(len(label)):
                for i in range(10):
                    if i == label[j] and predicted_label.argmax(1)[j] == label[j]:
                        acc[i] += 1
                        break
            total_count += label.size(0)
    acc = [x / (total_count / 4) for x in acc]
    return acc


if __name__ == '__main__':
    loss_temp = []
    path = 'AG_NEWS.data'  # 改成你数据集存放的路径
    train_iter = AG_NEWS(root=path, split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

    num_class = 4
    print(num_class)
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    # Hyper parameters
    EPOCHS = 25  # epoch
    LR = 0.1  # learning rate
    BATCH_SIZE = 64  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    total_accu = None
    train_iter, test_iter = AG_NEWS(root=path, split=('train', 'test'))
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_batch)

    acc_temp = []
    test_acc_temp = [[], [], [], []]
    test_acc = []
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        acc_temp.append(accu_val)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)
        test_acc = evaluate_test(test_dataloader)
        for i in range(len(test_acc_temp)):
            test_acc_temp[i].append(test_acc[i])
    print(test_acc)

    x = [i for i in range(1, len(acc_temp) + 1)]
    plt.plot(x, acc_temp, 'o-', color='g')  # o-:圆形
    plt.xlabel("Epoch")  # 横坐标名字
    plt.ylabel("ACC")  # 纵坐标名字
    plt.savefig('ACC.jpg')
    # plt.savefig('test_acc.jpg')
    plt.show()

    plt.plot(x, loss_temp, 'o-', color='g')  # o-:圆形
    plt.xlabel("Epoch")  # 横坐标名字
    plt.ylabel("Loss")  # 纵坐标名字
    plt.savefig('Loss.jpg')
    # plt.savefig('test_loss.jpg')
    plt.show()

    # 绘制每一类的准确率
    color = ['r', 'g', 'b', 'plum']
    ag_news_label = ["World", "Sports", "Business", "Sci/Tec"]
    for i in range(len(test_acc_temp)):
        plt.plot(x, test_acc_temp[i], '+-', color=color[i], label=ag_news_label[i])  # o-:圆形
    plt.xlabel("Epoch")  # 横坐标名字
    plt.ylabel("ACC")  # 纵坐标名字
    plt.legend()
    plt.savefig('ACC_class.jpg')
    # plt.savefig('test_loss.jpg')
    plt.show()

    torch.save(model.state_dict(), 'ChineseTextClassification.pth')
    # torch.save(model.state_dict(), 'test.pth')