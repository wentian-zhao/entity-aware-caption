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
        return " ".join(map(str, self.bpe.encode(x)))

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
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

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
        bpe_sentence = "<s> " + self.bpe.encode(sentence) + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )

        return tokens

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


if __name__ == '__main__':
    # roberta = RobertaModel.from_pretrained('../pretrained_model/roberta.large.tar.gz', 'model.pt')
    # roberta.task.source_dictionary.save('../data/roberta.large.dictionary')

    # tokenizer = RobertaTokenizer(encoder_json='../data/encoder.json', vocab_bpe='../data/vocab.bpe', dictionary_path='../data/roberta.large.dictionary')
    #
    # tokens = tokenizer.encode('...')
    # sent = tokenizer.decode(tokens)

    tokenizer = RobertaTokenizer(encoder_json='../data/encoder.json', vocab_bpe='../data/vocab.bpe',
                                 dictionary_path='../data/roberta.large.dictionary')
    tokens = tokenizer.encode('The quick brown fox jumps over a lazy dog')
