import torch
import numpy as np
import torch.nn.functional as F
import heapq
import copy
from nltk.tokenize import word_tokenize
import re





class BeamGenerator:
    def __init__(self, model, tokenizer, device='cuda', eos_token_id=3, max_seq_len=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, seed_text, max_steps_n=40, return_hypotheses_n=5, beamsize=5):

        seed_tokens = self.tokenizer.encode([seed_text])[0]
        initial_length = len(seed_tokens)
        partial_hypotheses = [(0, seed_tokens)]
        final_hypotheses = []

        while len(partial_hypotheses) > 0:
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)
            in_batch = torch.tensor(cur_partial_hypothesis).unsqueeze(0).to(self.device)

            seq_len = torch.tensor([len(cur_partial_hypothesis)]).to('cpu')
            next_tokens_logits = self.model(in_batch, lens=seq_len)[0, len(cur_partial_hypothesis) - 1]

            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=-1)
            topk_continuations = next_tokens_logproba.topk(beamsize, dim=-1)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                token_score = float(token_score)
                token_idx = int(token_idx)
                old_denorm_score = cur_partial_score * np.sqrt(len(cur_partial_hypothesis))
                new_score = (old_denorm_score - token_score) / np.sqrt(len(cur_partial_hypothesis) + 1)

                new_hypothesis = cur_partial_hypothesis
                if token_idx != 3:
                    new_hypothesis = cur_partial_hypothesis + [token_idx]
                new_item = (new_score, new_hypothesis)
                if (len(new_hypothesis) > self.max_seq_len):
                    break
                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= self.max_seq_len:
                    final_hypotheses.append(new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        final_texts = self.tokenizer.decode(list(final_token_lists))

        result = list(zip(final_scores, final_texts))
        result.sort()
        result = result[:return_hypotheses_n]

        return dict(result)


class GreedGenerator:

    def __init__(self, model, tokenizer, device='cpu', eos_token=3, pad_token=0, max_seq_len=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len

    def __call__(self, seed_phrase, n_samples=10):

        seed_seq = self.tokenizer.encode(seed_phrase)
        x_sequence = torch.tensor([self.tokenizer.encode(seed_phrase)]).to(self.device)
        seq_len = torch.tensor([len(seed_seq)]).to('cpu')

        predict = self.model(x_sequence, lens=seq_len)
        next_probs = F.softmax(predict, dim=-1)
        next_indexes = next_probs.topk(10, dim=-1).indices
        result = dict.fromkeys(range(0, n_samples - 1))

        next_indexes = next_indexes[0][len(seed_seq) - 1].detach().cpu().numpy()
        for idx, next_idx in enumerate(next_indexes):
            seq = copy.copy(seed_seq)

            if not (next_idx == self.eos_token or next_idx == self.pad_token):
                seq.append(next_idx)
                for i in range(self.max_seq_len):
                    if len(seq) >= self.max_seq_len:
                        break
                    seq_len = torch.tensor([len(seq)]).to('cpu')
                    in_batch = torch.tensor([seq]).to(self.device)
                    output = self.model(in_batch, lens=seq_len)
                    best_next_token = output[0, len(seq) - 1].argmax()

                    if best_next_token.item() == self.eos_token or best_next_token.item() == self.pad_token:
                        break
                    seq.append(best_next_token.item())
                decoded_seq = self.tokenizer.decode(seq)[0]
                result[idx] = decoded_seq
            else:
                decoded_seq = self.tokenizer.decode(seq)[0]
                result[idx] = decoded_seq

        return result


def find_words(word, vocab, max_words=5):
    top_words = []
    if word not in vocab:
        for key in vocab.keys():
            if len(top_words) == max_words:
                break
            if key[:len(word)] == word:
                top_words.append(key)
    else:
        return None
    return top_words

# def keyboard_regen(phrase, vocab):
#     alphabet = set("qwertyuiop[]asdfghjkl;'zxcvbnm,.``")
#     is_ok = alphabet.isdisjoint(phrase.lower())  # true if any eng_keyb sym
#
#     if is_ok:
#         return phrase
#     else:
#         fixed = []
#         regex = re.compile(r'[A-Za-z,.;\'``0-9]+|[А-Яа-я0-9]+|\d+')
#         phrase = regex.findall(phrase)
#         for word in phrase:
#             if not alphabet.isdisjoint(word):
#                 if word not in vocab:
#                     is_vocab = find_words(word, vocab)
#                     if is_vocab == []:
#                         word = fix_keyboard(word)
#             fixed.append(word)
#
#         return " ".join(fixed)
#
#
# def fix_keyboard(phrase):
#     rus_keyb = "йцукенгшщзхъфывапролджэячсмитьбюё"
#     eng_keyb = "qwertyuiop[]asdfghjkl;'zxcvbnm,.``"
#     fixed_phrase = ""
#     for symb in phrase:
#         idx = eng_keyb.find(symb)
#         if symb == " ":
#             fixed_phrase += " "
#         elif idx == -1:
#             continue
#         else:
#             fixed_phrase += rus_keyb[idx]
#     return fixed_phrase
