import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence


class TxtLSTM(nn.Module):
    def __init__(self, num_tokens=1000, seq_len=10, n_layers=2, embedding_size=200, rnn_num_units=300, embedding=None):
        super(self.__class__, self).__init__()
        self.seq_len = seq_len
        self.embedding = embedding
        if embedding == None:
            self.embedding = nn.Embedding(num_tokens, embedding_size, padding_idx=0)

        self.rnn = nn.LSTM(embedding_size, rnn_num_units, num_layers=n_layers, batch_first=True, dropout=0.5)
        self.bn = nn.BatchNorm1d(self.seq_len)
        self.hid_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, lens=None):
        # input: BATCH_SIZE x SEQ_LEN
        x = self.embedding(x)  # BATCH_SIZE x SEQ_LEN x EMB_SIZE
        x = pack_padded_sequence(x, lens, batch_first=True,
                                 enforce_sorted=False)  # data: NONZEROS_VALS X EMB_SIZE, batch_sizes: nonzeros_count in 1st seq symbol, 2nd etc.;
        # sorted and unsorted idxes: BATCH_SIZE

        h_seq, _ = self.rnn(x)  # data: NONZEROS_VALS X RNN_NUM_UNITS, batch_sizes: nonzeros_count in 1st seq symbol, 2nd etc.
        # sorted and unsorted idxes: BATCH_SIZE
        h_seq, seq_lens = pad_packed_sequence(h_seq, batch_first=True,
                                              total_length=self.seq_len)  # BATCH_SIZE x MAX_SEQ_LEN x RNN_NUM_UNITS
        h_seq = self.bn(h_seq)  # BATCH_SIZE x MAX_SEQ_LEN x RNN_NUM_UNITS
        next_logits = self.hid_to_logits(h_seq)  # BATCH_SIZE x SEQ_LEN x VOCAB_SIZE

        return next_logits


class TxtDataset(Dataset):
    def __init__(self, texts):
        self.texts = torch.tensor(texts, dtype=torch.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def train_model(model, epochs, optimizer, criterion, tr_dataloader, device='cpu'):
    model.to(device)
    model.train()
    history = []

    try:
        for i in range(epochs):
            n_ephoch_loss = []
            batch_ix = next(iter(tr_dataloader))
            batch_ix = batch_ix.to(device)

            pred = model(batch_ix)
            # compute loss
            pred = pred[:, :-1].permute(0, 2, 1)
            actual_next_tokens = batch_ix[:, 1:]
            loss = criterion(pred, actual_next_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            history.append(loss.cpu().data.numpy())
            loss_val = loss.cpu().data.numpy()
            n_ephoch_loss.append(loss_val)

            if (i + 1) % 10 == 0:
                # clear_output(True)
                # plt.plot(history, label='loss')
                # plt.legend()
                # plt.show()
                print("Mean Loss: " + str(np.mean(n_ephoch_loss)))

    except KeyboardInterrupt:
        print("Stopped by user!")
