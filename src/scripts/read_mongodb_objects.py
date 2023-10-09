"""
read goodnews and nytimes800k from mongodb
"""
import json
import os
import sys
from collections import namedtuple, defaultdict, Counter

import pymongo
import numpy as np
from PIL import Image
from tqdm import tqdm

import h5py

data_dir = os.path.join('..', 'data')
from config import image_dirs, object_and_face_dir

nytimes_image_dir = '/media/wentian/sdb1/BaiduNetdiskDownload/215629229_entalent/news_image_captioning/data/nytimes/images_processed'


mongo_host, mongo_port = '127.0.0.1', 27017
max_n_words = 500

IndexItem = namedtuple('IndexItem', field_names=('image_id', 'article_id', 'image_exists'))


def read_goodnews():
    client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
    db = client.goodnews

    goodnews_data_dir = os.path.join(data_dir, 'goodnews')
    if not os.path.exists(goodnews_data_dir):
        os.makedirs(goodnews_data_dir)
    goodnews_object_and_face_dir = os.path.join(object_and_face_dir, 'goodnews')
    if not os.path.exists(goodnews_object_and_face_dir):
        os.makedirs(goodnews_object_and_face_dir)
    goodnews_object_and_face_file = os.path.join(object_and_face_dir, 'object_and_face_goodnews.h5')
    f_h5 = h5py.File(goodnews_object_and_face_file, 'w')

    goodnews_image_dir = image_dirs['goodnews']

    # image_id, article_id, image_index, split
    sample_cursor = db.splits.find({}, projection=['_id', 'article_id', 'image_index', 'split', 'facenet_details'])

    goodnews_index = defaultdict(list)
    caption_dict = {}
    article_dict = {}

    image_exist_counter = Counter()

    for i, sample in tqdm(enumerate(sample_cursor), total=466385):
        image_id, article_id, image_index, split = (sample[key] for key in ('_id', 'article_id', 'image_index', 'split'))
        assert '{}_{}'.format(article_id, image_index) == image_id

        image_path = os.path.join(goodnews_image_dir, '{}.jpg'.format(image_id))

        image_exists = os.path.exists(image_path)
        image_exist_counter[image_exists] += 1
        goodnews_index[split].append(IndexItem(image_id, article_id, image_exists))

        article = db.articles.find_one({'_id': {'$eq': article_id}}, projection=['_id', 'context', 'images', 'web_url'])
        context = ' '.join(article['context'].strip().split(' ')[:max_n_words])
        caption = article['images'][image_index]
        caption = caption.strip()

        save_dict = {}
        objects = db.objects.find_one({'_id': image_id})
        if objects is not None:
            save_dict.update({
                'obj_feat': np.array(objects['object_features']),
                'obj_conf': np.array(objects['confidences']),
                'obj_class': np.array(objects['classes'], dtype=np.int16)
            })
        if 'facenet_details' in sample:
            faces = sample['facenet_details']
            save_dict.update({
                'face_feat': np.array(faces['embeddings']),
                'face_conf': np.array(faces['detect_probs'])
            })
        group = f_h5.create_group(image_id)
        for key, value in save_dict.items():
            if 'feat' in key:
                ds = group.create_dataset(name=key, shape=value.shape, dtype=value.dtype, compression="gzip")
            else:
                ds = group.create_dataset(name=key, shape=value.shape, dtype=value.dtype)
            ds[...] = value

        # if len(save_dict) > 0:
        #     f = os.path.join(goodnews_object_and_face_dir, '{}.pkl'.format(image_id))
        #     np.savez_compressed(f, **save_dict)

        caption_dict[image_id] = caption
        article_dict[article_id] = context

    # with open(os.path.join(goodnews_data_dir, 'index_goodnews.json'), 'w') as f:
    #     json.dump(goodnews_index, f)
    # with open(os.path.join(goodnews_data_dir, 'articles_goodnews.json'), 'w') as f:
    #     json.dump(article_dict, f)
    # with open(os.path.join(goodnews_data_dir, 'captions_goodnews.json'), 'w') as f:
    #     json.dump(caption_dict, f)

    print('splits:', dict((k, len(v)) for k, v in goodnews_index.items()))
    print('articles:', len(article_dict), 'captions:', len(caption_dict))
    print('image_exists:', image_exist_counter)


def read_nytimes():
    client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
    db = client.nytimes

    nytimes_data_dir = os.path.join(data_dir, 'nytimes')
    if not os.path.exists(nytimes_data_dir):
        os.makedirs(nytimes_data_dir)

    nytimes_object_and_face_file = os.path.join(object_and_face_dir, 'object_and_face_nytimes.h5')
    f_h5 = h5py.File(nytimes_object_and_face_file, 'w')

    sample_cursor = db.articles.find({}, projection=['_id'])
    id_list = []
    for item in tqdm(sample_cursor):
        id_list.append(item['_id'])

    # sample_cursor = db.articles.find({}, projection=['_id', 'parsed_section.type', 'parsed_section.text', 'parsed_section.hash', 'image_positions', 'headline', 'web_url', 'split'])

    nytimes_index = defaultdict(list)
    caption_dict = {}
    article_dict = {}

    image_exist_counter = Counter()
    duplicate_image_cnt = 0

    __i = 0
    # for article in tqdm(sample_cursor):
    for article_id in tqdm(id_list):
        article = db.articles.find_one({'_id': {'$eq': article_id}}, projection=[
            '_id', 'parsed_section.type', 'parsed_section.text', 'parsed_section.hash', 'image_positions', 'headline', 'web_url', 'split',
            'facenet_details'
        ])

        article_id = article['_id']
        sections = article['parsed_section']
        image_positions = article['image_positions']
        if 'split' not in article:
            __i += 1
            continue
        split = article['split']

        title = ''
        if 'main' in article['headline']:
            title = article['headline']['main'].strip()
        paragraphs = [s['text'].strip()
                      for s in sections if s['type'] == 'paragraph']
        if title:
            paragraphs.insert(0, title)
        n_words = 0
        for i, par in enumerate(paragraphs):
            n_words += len(par.split())
            if n_words > max_n_words:
                break

        context = '\n'.join(paragraphs[:i + 1]).strip()
        article_dict[article_id] = context

        for pos in image_positions:
            caption = sections[pos]['text'].strip()
            caption = caption.replace('\n', ' ').replace('\r', ' ')
            assert ('\n' not in caption) and ('\r' not in caption)
            if not caption:
                continue

            image_id = sections[pos]['hash']
            image_path = os.path.join(nytimes_image_dir, f"{image_id}.jpg")

            # if not os.path.exists(image_path):
            #     continue
            # try:
            #     image = Image.open(image_path)
            # except (FileNotFoundError, OSError):
            #     continue

            image_exists = os.path.exists(image_path)
            image_exist_counter[image_exists] += 1

            # TODO: check for duplicate image_id?
            nytimes_index[split].append(IndexItem(image_id, article_id, image_exists))
            caption_dict[image_id] = caption

            save_dict = {}
            objects = db.objects.find_one({'_id': {'$eq': image_id}})
            if objects is not None:
                save_dict.update({
                    'obj_feat': np.array(objects['object_features']),
                    'obj_conf': np.array(objects['confidences']),
                    'obj_class': np.array(objects['classes'], dtype=np.int16)
                })
            if 'facenet_details' in sections[pos]:
                faces = sections[pos]['facenet_details']
                save_dict.update({
                    'face_feat': np.array(faces['embeddings']),
                    'face_conf': np.array(faces['detect_probs'])
                })
            try:
                group = f_h5.create_group(image_id)
            except ValueError:
                duplicate_image_cnt += 1
                group = None

            if group is not None:
                for key, value in save_dict.items():
                    if 'feat' in key:
                        ds = group.create_dataset(name=key, shape=value.shape, dtype=value.dtype, compression="gzip")
                    else:
                        ds = group.create_dataset(name=key, shape=value.shape, dtype=value.dtype)
                    ds[...] = value

    # with open(os.path.join(nytimes_data_dir, 'index_nytimes.json'), 'w') as f:
    #     json.dump(nytimes_index, f)
    # with open(os.path.join(nytimes_data_dir, 'articles_nytimes.json'), 'w') as f:
    #     json.dump(article_dict, f)
    # with open(os.path.join(nytimes_data_dir, 'captions_nytimes.json'), 'w') as f:
    #     json.dump(caption_dict, f)

    print('no split:', __i)
    print('splits:', dict((k, len(v)) for k, v in nytimes_index.items()))
    print('articles:', len(article_dict), 'captions:', len(caption_dict))
    print('image_exists:', image_exist_counter)


if __name__ == '__main__':
    # read_goodnews()
    read_nytimes()