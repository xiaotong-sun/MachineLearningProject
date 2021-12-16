import torch
from torchtext.vocab import build_vocab_from_iterator
from train import TextClassificationModel
import jieba
import pandas as pd
import numpy as np


def tokenizer(text):
    return [word for word in jieba.cut(text) if word.strip()]


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


if __name__ == '__main__':
    train_iter = pd.read_csv('train.csv', encoding='utf-8')
    train_iter = np.array(train_iter)
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))

    device = torch.device("cpu")
    num_class = 10
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    model.load_state_dict(torch.load('model_TextClassification2.pth'))
    criterion = torch.nn.CrossEntropyLoss()

    ag_news_label = {1: "房产",
                     2: "股票",
                     3: "教育",
                     4: "科技",
                     5: "社会",
                     6: "时政",
                     7: "体育",
                     8: "游戏",
                     9: "娱乐",
                     10: "财经"}
    # ex_text_str = 'Beijing of Automation, Beijing Institute of Technology'
    ex_text_str = "组图：新《三国》再曝海量剧照 火战场面极震撼"

    print("这是一个 %s 新闻" % ag_news_label[predict(ex_text_str, text_pipeline)])