import json
import os
import pickle
import random
import sys
from collections import OrderedDict, Counter
from types import SimpleNamespace

import h5py
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta.alignment_utils import align_bpe_to_words
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

sys.path.append('.')

import torch
from fairseq.data import FairseqDataset, data_utils, Dictionary
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import regex as re

from config import data_dir, image_dirs, article_processed_dirs, roberta_path, object_and_face_dir
from util import RobertaTokenizer
from util.text_kg import read_article_kg, _get_words_from_article
from util.tokenizer_map import RobertaTokenizer as _RobertaTokenizer, _sublist_index, _rm_punct


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class NewsCaptionDataset(FairseqDataset):
    def __init__(self, args, dataset_name, dictionary, tokenizer, return_raw=False):
        self.args = args
        self.dataset_name = dataset_name
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        self.instances_per_epoch = args.instances_per_epoch

        self.use_text_graph = args.use_text_graph         # defined at class TransformerModelConvMM (transformer2_conv_mm)
        self.use_image_graph = args.use_image_graph
        self.return_raw = return_raw

        self.index_file = os.path.join(data_dir, dataset_name, 'index_{}.json'.format(dataset_name))        # created by read_mongodb.py
        self.captions_file = os.path.join(data_dir, dataset_name, 'captions_{}.json'.format(dataset_name))
        self.articles_file = os.path.join(data_dir, dataset_name, 'articles_{}.json'.format(dataset_name))

        self.article_preprocessed_dir = article_processed_dirs[dataset_name]

        # self.use_image = getattr(args, 'use_image', True)
        self.use_image = getattr(args, 'use_image', False)
        if self.use_image:
            self.image_dir = image_dirs[self.dataset_name]
            self.image_preprocess = Compose([
                # Resize(256), CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        with open(self.index_file, 'r') as f:
            index = json.load(f)                            # split -> (image_id, article_id)
        self.index = []                                     # index[i] = (image_id, article_id, split)
        for split, id_list in index.items():                # id_list[i] = (image_id, article_id, image_exists)
            id_list = filter(lambda x: x[2], id_list)
            self.index.extend((image_id, article_id, split) for (image_id, article_id, image_exists) in id_list)

        self.tokenization_cache_file = os.path.join(data_dir, dataset_name, 'tokenization_{}.pkl'.format(dataset_name))
        self.graph_cache_file = os.path.join(data_dir, dataset_name, 'article_graph_preprocessed_{}.pkl'.format(dataset_name))
        if not os.path.exists(self.tokenization_cache_file):
            self.tokenization_cache = None
        else:
            print('using token cache')
            with open(self.tokenization_cache_file, 'rb') as f:
                self.tokenization_cache = pickle.load(f)

        if self.use_text_graph:
            with open(self.graph_cache_file, 'rb') as f:
                self.graph_cache = pickle.load(f)
            # filter articles without graphs
            self.index = list(filter(lambda x: x[1] in self.graph_cache, self.index))

        if self.use_image_graph:
            self.f_obj = h5py.File(os.path.join(object_and_face_dir, 'object_and_face_{}.h5'.format(dataset_name)), 'r')

        self.train_index = [i for i in range(len(self.index)) if self.index[i][2] == 'train']

        with open(self.captions_file, 'r') as f:
            self.caption_dict = json.load(f)                 # image_id -> caption
        with open(self.articles_file, 'r') as f:
            self.article_dict = json.load(f)                 # article_id -> article


    def __getitem__(self, index):
        # always select training sample randomly
        # implement instances_per_epoch
        if self.instances_per_epoch > 0:
            _split = self.index[index][2]
            if _split == 'train':
                index = random.sample(self.train_index, k=1)[0]

        image_id, article_id = self.index[index][:2]
        caption = self.caption_dict[image_id].replace('\r', ' ').replace('\n', ' ').strip()
        article = self.article_dict[article_id]

        if self.tokenization_cache is not None:
            src_tokens = self.tokenization_cache['articles'][article_id]
            tgt_tokens = self.tokenization_cache['captions'][image_id]
            src_tokens, tgt_tokens = torch.LongTensor(src_tokens), torch.LongTensor(tgt_tokens)
        else:
            src_tokens = self.tokenizer.encode(article).to(torch.long)      # [bos_idx, ......, eos_idx]
            tgt_tokens = self.tokenizer.encode(caption).to(torch.long)      # [bos_idx, ...., eos_idx]

        if len(src_tokens) > 512:
            src_tokens = torch.cat((src_tokens[:1], src_tokens[1:511], src_tokens[-1:]), dim=0)
        if len(tgt_tokens) > 50:
            tgt_tokens = torch.cat((tgt_tokens[:1], tgt_tokens[1:49], tgt_tokens[-1:]), dim=0)

        # src_tokens = tgt_tokens

        d = {
            'id': index,
            'image_id': image_id,
            'article_id': article_id,
            'src_tokens': src_tokens,
            'tgt_tokens': tgt_tokens,
        }

        if self.use_image:
            image_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
            image = Image.open(image_path)
            image = self.image_preprocess(image)
            d.update({
                'image': image
            })

        if self.return_raw:
            d.update({
                'article_raw': article,
                'caption_raw': caption,
            })

        if self.use_text_graph:
            article_graph = self.graph_cache[article_id]
            node_bpe_index, edges = article_graph['node_bpe_index'], article_graph['edge']
            d.update({
                'node_bpe_index': node_bpe_index,
                'edges': edges
            })

        if self.use_image_graph:
            feat_dict = self.f_obj[image_id]
            if 'obj_feat' in feat_dict:
                obj_feat = np.array(feat_dict['obj_feat'])      # (? * 2048)
            else:
                obj_feat = np.zeros((0, 2048))
            if 'face_feat' in feat_dict:
                face_feat = np.array(feat_dict['face_feat'])    # (? * 512)
            else:
                face_feat = np.zeros((0, 512))
            d.update({
                'obj_feat': obj_feat, 'face_feat': face_feat
            })

        return d

    def __len__(self):
        return len(self.index)

    def collater(self, samples):
        def _batch_field(key):
            return [sample[key] for sample in samples]

        batch_size = len(samples)

        samples.sort(key=lambda x: len(x['src_tokens']), reverse=True)

        d = {
            'id': np.array(_batch_field('id')),
            'image_id': _batch_field('image_id'),
            'article_id': _batch_field('article_id'),
        }

        src_tokens_batch = _batch_field('src_tokens')
        tgt_tokens_batch = _batch_field('tgt_tokens')
        # tgt_tokens_batch = [i[1:] for i in tgt_tokens_batch]      # remove <eos> at the beginning

        net_input = {
            'src_tokens': data_utils.collate_tokens(src_tokens_batch, pad_idx=self.dictionary.pad_index, eos_idx=self.dictionary.eos_index, move_eos_to_beginning=False),
            'src_lengths': torch.LongTensor([len(sample['src_tokens']) for sample in samples]),
            'prev_output_tokens': data_utils.collate_tokens(tgt_tokens_batch, pad_idx=self.dictionary.pad_index, eos_idx=self.dictionary.eos_index, move_eos_to_beginning=True),
        }
        # other features here
        if self.use_image:
            net_input.update({
                'image': torch.stack([sample['image'] for sample in samples], dim=0)
            })

        if self.use_text_graph:
            net_input.update({
                'node_bpe_index': _batch_field('node_bpe_index'),
                'edges': _batch_field('edges')
            })

        if self.use_image_graph:
            obj_feat_batch = []
            obj_feat_index = []
            face_feat_batch = []
            face_feat_index = []
            for sample in samples:
                obj_feat = sample['obj_feat']
                s, e = len(obj_feat_batch), len(obj_feat_batch) + len(obj_feat)
                obj_feat_index.append((s, e))
                obj_feat_batch.extend(obj_feat)

                face_feat = sample['face_feat']
                s, e = len(face_feat_batch), len(face_feat_batch) + len(face_feat)
                face_feat_index.append((s, e))
                face_feat_batch.extend(face_feat)

            if len(obj_feat_batch) > 0:
                obj_feat_batch = np.stack(obj_feat_batch, axis=0)
            else:
                obj_feat_batch = np.zeros(shape=(0, 2048))
            if len(face_feat_batch) > 0:
                face_feat_batch = np.stack(face_feat_batch, axis=0)
            else:
                face_feat_batch = np.zeros(shape=(0, 512))

            net_input.update({
                'obj_feat': torch.Tensor(obj_feat_batch),             # (total_feats, 2048)
                'obj_feat_index': torch.LongTensor(np.array(obj_feat_index)),       # (batch_size, 2)
                'face_feat': torch.Tensor(face_feat_batch),           # (total_feats, 512)
                'face_feat_index': torch.LongTensor(np.array(face_feat_index))      # (batch_size, 2)
            })

        d['net_input'] = net_input

        d.update({
            'target': data_utils.collate_tokens(tgt_tokens_batch, pad_idx=self.dictionary.pad_index, eos_idx=self.dictionary.eos_index, move_eos_to_beginning=False),
            'ntokens': sum([len(sample['src_tokens']) for sample in samples] + [len(sample['tgt_tokens']) for sample in samples]),
            'nsentences': batch_size,
        })
        return d

    def num_tokens(self, index):
        image_id, article_id = self.index[index][:2]
        return len(self.article_dict[article_id].split()) + len(self.caption_dict[image_id].split())

    def size(self, index):
        image_id, article_id = self.index[index][:2]
        return min(len(self.article_dict[article_id].split()), 512), min(len(self.caption_dict[image_id].split()), 51)
        # return len(self.article_dict[article_id].split()), len(self.caption_dict[image_id].split())

    def prefetch(self, indices):
        pass

    def get_split_index(self, split, **kwargs):
        print(f'-------- get_split_index {split} --------')

        if split == 'test-small':
            split = 'test'
            small = True
        else:
            small = False

        split_index = [i for i, item in enumerate(self.index) if item[2] == split]

        # ==== always select training samples randomly
        # implement instances_per_epoch
        if self.instances_per_epoch > 0:
            if split == 'train':
                split_index = random.sample(split_index, k=self.instances_per_epoch)

        if small:
            split_index = split_index[:10000]

        return split_index

    def get_gt_sent_by_image_id(self, image_id):
        gt_sent = self.caption_dict[image_id]
        return gt_sent.replace('\n', ' ').replace('\r', ' ')


class FairseqDatasetSubset(FairseqDataset):
    def __init__(self, dataset, index_list):
        self.dataset = dataset
        self.index_list = index_list

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.index_list[index])

    def __len__(self):
        return len(self.index_list)

    def collater(self, samples):
        return self.dataset.collater(samples)

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.index_list[index])

    def size(self, index):
        return self.dataset.size(self.index_list[index])

    def prefetch(self, indices):
        _indices = [self.index_list[i] for i in indices]
        return self.dataset.prefetch(_indices)


def gen_tokenization_cache(dataset_name):
    dictionary = Dictionary.load(os.path.join(data_dir, 'roberta.large.dictionary'))
    tokenizer = RobertaTokenizer(
        encoder_json=os.path.join(data_dir, 'encoder.json'),
        vocab_bpe=os.path.join(data_dir, 'vocab.bpe'),
        dictionary_path=os.path.join(data_dir, 'roberta.large.dictionary')
    )
    args = SimpleNamespace()
    setattr(args, 'instances_per_epoch', 800000)
    setattr(args, 'use_text_graph', 0)
    setattr(args, 'use_image_graph', 0)

    dataset = NewsCaptionDataset(args, dataset_name, dictionary=dictionary, tokenizer=tokenizer)

    if os.path.exists(dataset.tokenization_cache_file):
        with open(dataset.tokenization_cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {'articles': {}, 'captions': {}}

    from tqdm import tqdm
    for article_id, caption in tqdm(dataset.article_dict.items(), total=len(dataset.article_dict), ncols=80):
        tokens = tokenizer.encode(caption).numpy()
        cache['articles'][article_id] = tokens
    for image_id, caption in tqdm(dataset.caption_dict.items(), total=len(dataset.caption_dict), ncols=80):
        caption = caption.replace('\r', ' ').replace('\n', ' ').strip()
        if '\r' in caption or '\n' in caption:
            print('image id: {}, sent: {}'.format(image_id, caption))
        tokens = tokenizer.encode(caption).numpy()
        cache['captions'][image_id] = tokens

    with open(dataset.tokenization_cache_file, 'wb') as f:
        pickle.dump(cache, f)


def get_graph_cache(dataset_name):
    dictionary = Dictionary.load(os.path.join(data_dir, 'roberta.large.dictionary'))
    tokenizer = RobertaTokenizer(
        encoder_json=os.path.join(data_dir, 'encoder.json'),
        vocab_bpe=os.path.join(data_dir, 'vocab.bpe'),
        dictionary_path=os.path.join(data_dir, 'roberta.large.dictionary')
    )
    args = SimpleNamespace()
    setattr(args, 'instances_per_epoch', 800000)
    setattr(args, 'use_text_graph', 0)
    setattr(args, 'use_image_graph', 0)

    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    _tokenizer = _RobertaTokenizer(
        encoder_json=os.path.join(data_dir, 'encoder.json'),
        vocab_bpe=os.path.join(data_dir, 'vocab.bpe'),
        dictionary_path=os.path.join(data_dir, 'roberta.large.dictionary')
    )

    roberta = RobertaModel.from_pretrained(roberta_path, checkpoint_file='model.pt')

    token_counter = Counter()
    entity_counter = Counter()
    wiki_counter = Counter()

    article_graph_cache = dict()

    dataset = NewsCaptionDataset(args, dataset_name, dictionary=dictionary, tokenizer=tokenizer, return_raw=True)
    # for instance_index, data in tqdm(enumerate(dataset), total=len(dataset)):

    for instance_index, (article_id, article) in tqdm(enumerate(dataset.article_dict.items()), total=len(dataset.article_dict), ncols=80):
        p_article_path = os.path.join(dataset.article_preprocessed_dir, '{}.json'.format(article_id))
        if not os.path.exists(p_article_path):      # without graph
            continue
        with open(p_article_path, 'r') as f:
            d_article = json.load(f)

        article_bpe_tokens, indexes, article_words = _tokenizer.encode(article)

        graph = read_article_kg(d_article)
        sents = [' '.join(_[0] for _ in sentence['tokens']) for sentence in d_article['sentences']]

        # _bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in article_bpe_tokens]
        # _bpe_tokens = [(roberta.bpe.decode(x) if x not in {"<s>", ""} else x).strip() for x in _bpe_tokens]
        # _s_bpe_tokens = ''.join(_bpe_tokens)
        # _s_corenlp_tokens = ""
        # corenlp_tokens = []
        # corenlp_sent_len_prefix = []
        # _s = 0
        # for i, sentence in enumerate(d_article['sentences']):
        #     corenlp_sent_len_prefix.append(_s); _s += len(sentence['tokens'])
        #     for t in sentence['tokens']:
        #         _s_corenlp_tokens += t[0].strip()
        #         if len(_s_corenlp_tokens) > len(_s_bpe_tokens):
        #             break
        #         corenlp_tokens.append(t[0])

        _article = '\n'.join(sents)

        # _article_tokens, _indexes, _article_words = _tokenizer.encode(_article)
        # _article_words = [i[1:] if i[0] == ' ' else i for i in _article_words]
        # aligned = align_bpe_to_words(roberta, article_bpe_tokens, corenlp_tokens)

        article_words = [i[1:] if i[0] == ' ' else i for i in article_words]
        node_words = [_get_words_from_article(d_article, span) for span in graph.nodes_list]
        node_entities = [_get_words_from_article(d_article, span, type='entity') for span in graph.nodes_list]
        node_wikis = [_get_words_from_article(d_article, span, type='wiki') for span in graph.nodes_list]

        removed_node_index_list = []
        node_bpe_index_list = []
        node_entity_type_list = []
        node_wiki_list = []

        for i, phrase in enumerate(node_words):
            span = graph.nodes_list[i]
            entity_type, wiki = node_entities[i], node_wikis[i]

            _tokens = list(re.findall(pat, phrase))
            _tokens = [i[1:] if i[0] == ' ' and i[1] != '\'' else i for i in _tokens]
            tokens = []
            for t in _tokens:
                if t[0] == '\'':
                    tokens.extend(['\'', t[1:]])
                else:
                    tokens.append(t)

            # the index of tokens (words) in the tokens (words) of the article
            _token_index = _sublist_index(article_words, tokens)

            if _token_index is not None and _token_index[1] < len(article_words):
                _bpe_index = indexes.index(_token_index[0]), indexes.index(_token_index[1])
                # __index = _sublist_index(_article_words, tokens)

                # article_words[_bpe_index[0] : _bpe_index[1]] == tokens
                _restored_phrase = _tokenizer.decode(article_bpe_tokens[_bpe_index[0] : _bpe_index[1]])
                if _rm_punct(_restored_phrase) != _rm_punct(phrase):
                    print('{} -> {}'.format(phrase, _restored_phrase))
                    import IPython; IPython.embed()

                if _bpe_index[1] >= 512:
                    _bpe_index = None
            else:
                _bpe_index = None

            if _bpe_index is None:
                token_counter[False] += 1
                removed_node_index_list.append(i)
            else:
                token_counter[True] += 1
                entity_counter[entity_type is not None] += 1
                wiki_counter[wiki is not None] += 1
                node_bpe_index_list.append(_bpe_index)
                node_entity_type_list.append(entity_type)
                node_wiki_list.append(wiki)

            # sent_index = span[0]
            # _bpe_index = None
            #
            # word_index_start = corenlp_sent_len_prefix[sent_index] + span[1][0]
            # word_index_end = corenlp_sent_len_prefix[sent_index] + span[1][1] - 1  # the word at the end of the phrase
            # if word_index_end < len(aligned):
            #     bpe_index_start = aligned[word_index_start][0]
            #     bpe_index_end = aligned[word_index_end][-1] + 1
            #     _bpe_index = bpe_index_start, bpe_index_end
            #
            #     if _rm_punct(' '.join(corenlp_tokens[word_index_start : word_index_end + 1])) != _rm_punct(tokenizer.decode(article_bpe_tokens[bpe_index_start: bpe_index_end])):
            #         print(' '.join(corenlp_tokens[word_index_start: word_index_end + 1]), '-', tokenizer.decode(article_bpe_tokens[bpe_index_start: bpe_index_end]))
            #         import IPython; IPython.embed()
            #
            # if _bpe_index is None:
            #     token_counter[False] += 1
            #     removed_node_index_list.append(i)
            # else:
            #     token_counter[True] += 1
            #     entity_counter[entity_type is not None] += 1
            #     wiki_counter[wiki is not None] += 1
            #     node_bpe_index_list.append(_bpe_index)
            #     node_entity_type_list.append(entity_type)
            #     node_wiki_list.append(wiki)

        graph.remove_nodes(removed_node_index_list)

        assert len(graph.nodes_list) <= 32767
        assert len(article_words) <= 32767

        # article_graph_cache[article_id] = {'node_bpe_index': node_bpe_index_list, 'edge': graph.edges}
        _node_bpe_index = np.array(node_bpe_index_list, dtype=np.int16)
        _edge = np.array(list(graph.edges), dtype=np.int16)

        for _e in _edge:
            u, v = _node_bpe_index[_e[0]], _node_bpe_index[_e[1]]
            _u, _v = _tokenizer.decode(article_bpe_tokens[u[0] : u[1]]), _tokenizer.decode(article_bpe_tokens[v[0] : v[1]])
            # print('edge:', _u, '->', _v)

        article_graph_cache[article_id] = {'node_bpe_index': _node_bpe_index, 'node_entity_type': node_entity_type_list,
                                           'node_wiki': node_wiki_list, 'edge': _edge}

        if instance_index % 10000 == 0:
            print('token:', token_counter, 'entity:', entity_counter, 'wiki:', wiki_counter)

    with open(dataset.graph_cache_file, 'wb') as f:
        pickle.dump(article_graph_cache, f)


if __name__ == '__main__':
    # gen_tokenization_cache('goodnews')
    # gen_tokenization_cache('goodnews')

    get_graph_cache('goodnews')
    get_graph_cache('nytimes')
