import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

from train import TextClassificationModel


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


if __name__ == '__main__':
    path = 'AG_NEWS.data'  # 改成你数据集存放的路径
    train_iter = AG_NEWS(root=path, split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = 4
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    model.load_state_dict(torch.load('model_TextClassification.pth'))
    criterion = torch.nn.CrossEntropyLoss()

    ag_news_label = {1: "World",
                     2: "Sports",
                     3: "Business",
                     4: "Sci/Tec"}

    # ex_text_str = 'Beijing of Automation, Beijing Institute of Technology'
    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was enduring the season’s worst weather conditions on " \
                  "Sunday at The Open on his way to a closing 75 at Royal Portrush, which considering the wind and " \
                  "the rain was a respectable showing. Thursday’s first round at the WGC-FedEx St. Jude Invitational " \
                  "was another story. With temperatures in the mid-80s and hardly any wind, the Spaniard was 13 " \
                  "strokes better in a flawless round. Thanks to his best putting performance on the PGA Tour, " \
                  "Rahm finished with an 8-under 62 for a three-stroke lead, which was even more impressive " \
                  "considering he’d never played the front nine at TPC Southwind. "

    print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])