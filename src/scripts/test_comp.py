import os
import sys
import json
import csv
from pymongo import MongoClient

client = MongoClient(host='127.0.0.1', port=27017)
db = client.nytimes

# index_file = r'/media/wentian/sdb1/work/news_caption_fairseq/data/goodnews/index_goodnews.json'
# ann_file = r'/media/wentian/sdb1/work/news_caption_fairseq/data/person_split_goodnews.json'
# comp_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer2_conv_mm_image/validation_results_test-small/compare_34_50.csv'

index_file = r'/media/wentian/sdb1/work/news_caption_fairseq/data/nytimes/index_nytimes.json'
ann_file = r'/media/wentian/sdb1/work/news_caption_fairseq/data/person_split_nytimes.json'
comp_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_nytimes/validation_results_test-small/compare_38_20.csv'


with open(index_file, 'r') as f:
    d = json.load(f)
    image_id_to_article = dict(i[:2] for i in d['test'])

with open(ann_file, 'r') as f:
    split = json.load(f)
no_person = set(split['no_person'])

with open(comp_file, 'r') as f:
    lines = []
    for i, line in enumerate(f):
        if i == 0: lines.append(line)
        else:
            image_id = line[:line.index(',')]
            if image_id in no_person:
                article_id = image_id_to_article[image_id]
                article = db.articles.find_one({
                    '_id': {'$eq': article_id},
                }, projection=['_id', 'context', 'images', 'web_url'])
                url = article['web_url']

                lines.append(line.rstrip() + ',' + url + '\n')

with open(r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_nytimes/validation_results_test-small/compare_38_20_filter.csv', 'w') as f:
    f.writelines(lines)
