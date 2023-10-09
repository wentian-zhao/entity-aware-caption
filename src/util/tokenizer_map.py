import json
import os
import pickle
import sys

import numpy as np
import torch
from fairseq import hub_utils
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder, bytes_to_unicode, get_pairs
from fairseq.models.roberta import RobertaModel, encoders
from fairseq.data.dictionary import Dictionary

import regex as re


class _GPT2BPE:
    """
    placeholder of GPT2BPE that contains fairseq.data.encoders.gpt2_bpe_utils.Encoder
    """
    def __init__(self, bpe):
        self.bpe = bpe

    def encode(self, x: str) -> str:
        enc, indexes, words = self.bpe.encode(x)
        return " ".join(map(str, enc)), indexes, words

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")


# # fairseq.data.encoders.gpt2_bpe_utils.Encoder
class _Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # try:
        #     import regex as re
        #
        #     self.re = re
        # except ImportError:
        #     raise ImportError("Please install regex with: pip install regex")

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        token_indexes = []
        words = []
        for token_index, token in enumerate(re.findall(self.pat, text)):
            words.append(token)
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            _bpe_token = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")]
            bpe_tokens.extend(_bpe_token)
            token_indexes.extend([token_index] * len(_bpe_token))

        return bpe_tokens, token_indexes, words

    def decode(self, tokens):
        text = "".join([self.decoder.get(token, token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text


# fairseq.data.encoders.gpt2_bpe_utils
def _get_encoder(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, "r") as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return _Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


class RobertaTokenizer:
    def __init__(self, encoder_json, vocab_bpe, dictionary_path):
        bpe = _get_encoder(encoder_json, vocab_bpe)  # fairseq.data.encoders.gpt2_bpe_utils.Encoder
        self.bpe = _GPT2BPE(bpe)
        self.source_dictionary = Dictionary.load(dictionary_path)

    def encode(self, sentence: str, *addl_sentences, no_separator=False):
        _bpe_sentence, indexes, words = self.bpe.encode(sentence)
        bpe_sentence = "<s> " + _bpe_sentence + " </s>"
        indexes = [-1] + indexes + [-1]
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            if not no_separator:
                indexes.append(-1)

            _bpe_sentence, _indexes = self.bpe.encode(s)
            bpe_sentence += " " + _bpe_sentence + " </s>"
            indexes.extend(_indexes)
            indexes.append(-1)
        tokens = self.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )

        return tokens, indexes, words

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences


article = """Roger Furst, 78, an Outfitter For Northeast Adventurers

Roger Furst, one of two fishing buddies who founded the Eastern Mountain Sports retail chain, which has equipped thousands of people to trek deep into the mountains, ski in the backcountry or scale a peak, died on March 16 in Sherman, Tex., a day after his 78th birthday.

His death was confirmed by his son Jason, who did not specify a cause.

Mr. Furst was a young lawyer in Denver in the early 1960s when he met Alan McDonough, a local hotel manager, and the two began hiking and fishing along the streams of the Colorado Rockies.

"We got tired of walking two to five miles back to our car in the dark," Mr. Furst told Backpacker magazine in 1974. "We started carrying our camp on our backs so we could sleep over and cook our fish right away."

But the bulkiness of the equipment prompted Mr. Furst and Mr. McDonough to begin searching for manufacturers of lightweight backpacks, hiking boots, sleeping bags and camp stoves. Realizing that there was a growing market for such equipment, they decided to shift careers and open their first store -- but not in the West.

With far fewer mountaineering stores in the Northeast, they decided that Wellesley, Mass., near Boston, was the right place to open what in 1967 was called the Mountain Shop. Demand was immediate, and within a year Eastern Mountain Sports had a 10,000-square-foot store in Boston, the largest outdoor equipment outlet in New England. That year, the partners also opened the EMS Climbing School in North Conway, N.H., now the oldest rock-climbing school in the Eastern United States.

"They helped shape the industry," said Will Manzer, the current chief executive of Eastern Mountain Sports. "They bought ice-climbing gear, mountain-climbing gear, rock-climbing gear, backpacking and backcountry skiing equipment from Europe and in the United States. And ultimately, in the '70s, we started to make our own equipment."
    """

_article = """Roger Furst , 78 , an Outfitter For Northeast Adventurers Roger Furst , one of two fishing buddies who founded the Eastern Mountain Sports retail chain , which has equipped thousands of people to trek deep into the mountains , ski in the backcountry or scale a peak , died on March 16 in Sherman , Tex. , a day after his 78th birthday .
His death was confirmed by his son Jason , who did not specify a cause .
Mr. Furst was a young lawyer in Denver in the early 1960s when he met Alan McDonough , a local hotel manager , and the two began hiking and fishing along the streams of the Colorado Rockies .
" We got tired of walking two to five miles back to our car in the dark , " Mr. Furst told Backpacker magazine in 1974 .
" We started carrying our camp on our backs so we could sleep over and cook our fish right away . "
But the bulkiness of the equipment prompted Mr. Furst and Mr. McDonough to begin searching for manufacturers of lightweight backpacks , hiking boots , sleeping bags and camp stoves .
Realizing that there was a growing market for such equipment , they decided to shift careers and open their first store -- but not in the West .
With far fewer mountaineering stores in the Northeast , they decided that Wellesley , Mass. , near Boston , was the right place to open what in 1967 was called the Mountain Shop .
Demand was immediate , and within a year Eastern Mountain Sports had a 10,000 - square - foot store in Boston , the largest outdoor equipment outlet in New England .
That year , the partners also opened the EMS Climbing School in North Conway , N.H. , now the oldest rock - climbing school in the Eastern United States .
" They helped shape the industry , " said Will Manzer , the current chief executive of Eastern Mountain Sports .
" They bought ice - climbing gear , mountain - climbing gear , rock - climbing gear , backpacking and backcountry skiing equipment from Europe and in the United States .
And ultimately , in the '70s , we started to make our own equipment .
\""""

node_words = ['ski', 'in', 'backcountry', 'Roger Furst', '78', 'Sherman', 'day after', 'his 78th birthday',
              'Outfitter For', 'Northeast Adventurers Roger Furst', 'one of', 'two fishing buddies', 'mountains',
              'ski in', 'his', 'son', 'Jason', 'His death', 'was confirmed by', 'Alan McDonough', 'began',
              'hiking and fishing along the streams of the Colorado Rockies', 'was',
              'young lawyer in Denver in the early 1960s', 'hiking', 'streams of the Colorado Rockies', 'lawyer', 'in',
              '1960s', 'in', 'Denver', 'lawyer in', 'Furst', 'was a young', 'was a young lawyer', 'early 1960s',
              'was a young lawyer in', 'told', 'Backpacker magazine', 'told Backpacker magazine in', '1974',
              'the Colorado Rockies', 'cook', 'our fish', 'carrying', 'our camp', 'right away', 'Mr. McDonough',
              'searching', 'hiking boots', 'manufacturers', 'camp stoves', 'bulkiness', 'prompted', 'Mr. Furst',
              'bulkiness of', 'equipment', 'fewer mountaineering stores', 'in', 'Northeast',
              'Mr. Furst and Mr. McDonough', 'decided', 'far fewer mountaineering stores in the Northeast',
              'Wellesley , Mass.', 'was', 'to open', 'was the right place', 'Eastern Mountain Sports', 'had',
              'largest outdoor equipment outlet in New England', 'Boston', 'in', 'largest outdoor equipment outlet',
              'in', 'England', 'Demand', 'was', 'immediate', '10,000 - square - foot store', 'outlet in',
              'EMS Climbing School', 'in', 'Conway', 'oldest rock - climbing school', 'in', 'United States',
              'school in', 'partners', 'opened the EMS', 'year', 'also opened the EMS', 'Will Manzer', 'executive of',
              'climbing gear', 'Europe', 'backpacking', 'bought', 'ice - climbing gear', 'make', 'our own equipment']


import string

def _rm_punct(s):       # remove punctuations and space
    return s.translate(str.maketrans('', '', string.punctuation + " \n"))

def _sublist_index(l, sl, start_index=0):
    # results = []

    sll = len(sl)

    _sl_str = _rm_punct(''.join(sl))

    for ind in (i for i, e in enumerate(l[start_index:]) if e == sl[0]):
        sl2 = 1
        while sl2 <= sll:
            p = _rm_punct(''.join(l[ind : ind + sl2]))
            if not (p in _sl_str):
                break
            if p == _sl_str:
                return (ind, ind + sl2)
            sl2 += 1

        # if l[ind:ind + sll] == sl:
        #     return (ind, ind + sll)
        # if ''.join(l[ind : ind + sll - 1]).replace(' .,!?', '') == _sl_str:
        #     return (ind, ind + sll - 1)

    return None


if __name__ == '__main__':
    # roberta = RobertaModel.from_pretrained('../pretrained_model/roberta.large.tar.gz', 'model.pt')
    # roberta.task.source_dictionary.save('../data/roberta.large.dictionary')

    # tokenizer = RobertaTokenizer(encoder_json='../data/encoder.json', vocab_bpe='../data/vocab.bpe', dictionary_path='../data/roberta.large.dictionary')
    #
    # tokens = tokenizer.encode('...')
    # sent = tokenizer.decode(tokens)

    tokenizer = RobertaTokenizer(encoder_json='../data/encoder.json', vocab_bpe='../data/vocab.bpe',
                                 dictionary_path='../data/roberta.large.dictionary')
    # tokens, indexes = tokenizer.encode('The quick brown fox jumps over a lazy dog')

    tokens, indexes, words = tokenizer.encode(_article)
    print(tokens)
    print(indexes)
    print(words)
    words = [i[1:] if i[0] == ' ' else i for i in words]

    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    for phrase in node_words:
        tokens = list(re.findall(pat, phrase))
        tokens = [i[1:] if i[0] == ' ' else i for i in tokens]
        index = _sublist_index(words, tokens)
        print(tokens, index)
