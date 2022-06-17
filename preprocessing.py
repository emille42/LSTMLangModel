# import numpy as np
import collections
from nltk.tokenize import word_tokenize
import re
import youtokentome as yttm


def build_vocab(queries, min_wrd_count: int = 0, pad_sym: str = None):
    word_counts = collections.defaultdict(int)

    for query in queries:
        for word in word_tokenize(query):
            word_counts[word] += 1
    if min_wrd_count > 0:
        word_counts = {word: count for word, count in word_counts.items() if count > min_wrd_count}

    vocab = {word: idx for idx, word in enumerate(list(word_counts.keys()))}
    if pad_sym is not None:
        vocab[pad_sym] = len(vocab)

    return vocab, word_counts


# работает с pd колонкой (вместе с apply), обработка по регулярному выражению
def process_text(query, min_q_len=1, max_q_len=7, min_word_len=2, regexp=r'[A-Za-z0-9]+|[А-Яа-я]+|\d+'):
    regex = re.compile(regexp)
    query = regex.findall(query)
    query = [word.lower() for word in query if len(word) > min_word_len]
    query = ' '.join(query)
    if len(word_tokenize(query)) > min_q_len and len(word_tokenize(query)) < max_q_len:
        return query
    else:
        return None


# возвращает натренированный токенизатор
def train_bpe(q_column, train_data_path=None, model_path="bpe.model", vocab_size=1500):
    # предобработка датасета (колонка с запросами)
    q_column = q_column.apply(process_text)
    q_column = q_column.dropna()
    q_column.to_csv(train_data_path, index=False)
    # тренировка BPE и сохранение модели по указанному пути
    if train_data_path is not None:
        yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=model_path)
    # загрузка модели по указанному пути
    bpe = yttm.BPE(model=model_path)
    return bpe
