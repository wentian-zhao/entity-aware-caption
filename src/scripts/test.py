import os
import sys
import pickle

import torch
from fairseq import hub_utils
from fairseq.models.roberta import RobertaModel, encoders
from fairseq.data.dictionary import Dictionary

from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.model_parallel.models.roberta.model import ModelParallelRobertaModel


sents = [
    'Byte Pair Encoding is Suboptimal for Language Model Pretraining',
    'Byte Pair Encoding',
    'byte pair encoding',
]


# roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.large')
roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')


roberta.task.source_dictionary.save('../data/roberta.base.dictionary')

x = hub_utils.from_pretrained(
            'roberta.base',
            'model.pt',
            '.',
            archive_map=RobertaModel.hub_models(),
            bpe='gpt2',
            load_checkpoint_heads=True,
)
args = x['args']
bpe = encoders.build_bpe(args)
source_dictionary = Dictionary.load('../data/roberta.base.dictionary')


def encode(bpe, source_dictionary, sentence: str, *addl_sentences, no_separator=False) -> torch.LongTensor:
    """
    BPE-encode a sentence (or multiple sentences).

    Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
    Every sentence ends with an end-of-sentence (`</s>`) and we use an
    extra end-of-sentence (`</s>`) as a separator.

    Example (single sentence): `<s> a b c </s>`
    Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

    The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
    requires leading spaces. For example::

        >>> roberta.encode('Hello world').tolist()
        [0, 31414, 232, 2]
        >>> roberta.encode(' world').tolist()
        [0, 232, 2]
        >>> roberta.encode('world').tolist()
        [0, 8331, 2]
    """
    bpe_sentence = "<s> " + bpe.encode(sentence) + " </s>"
    for s in addl_sentences:
        bpe_sentence += " </s>" if not no_separator else ""
        bpe_sentence += " " + bpe.encode(s) + " </s>"
    tokens = source_dictionary.encode_line(
        bpe_sentence, append_eos=False, add_if_not_exist=False
    )
    return tokens.long()


for sent in sents:
    tokens = roberta.encode(sent)
    print('sent:', sent, '\n', 'tokens:', tokens)
    print('tokens:', encode(bpe, source_dictionary, sent))
    sent1 = roberta.decode(tokens)


import torch.nn as nn
from torchvision.models import resnet152

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet152()   # pre-trained model to be excluded

    def forward(self, x):
        return self.resnet(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = ...          # some other network

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)  # feature extraction only
        y = self.decoder(feat)      # do something else
        return y