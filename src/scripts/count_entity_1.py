import os
import sys
from tqdm import tqdm
import json
from collections import Counter

caption_preprocessed_dir = '/media/wentian/nvme1/article_data/caption_preprocessed/goodnews/'
article_preprocessed_dir = '/media/wentian/nvme1/article_data/preprocessed/goodnews/'


def get_entities(preprocessed_dir):
    files = os.listdir(preprocessed_dir)

    linked_entities = Counter()

    for file in tqdm(files, ncols=80):
        f = open(os.path.join(preprocessed_dir, file), 'r')
        d = json.load(f)
        for sent in d['sentences']:
            for token in sent['tokens']:            # token: # (tokens, type, wiki)
                if len(token) >= 3:
                    linked_entities[(token[2], token[1])] += 1

    return linked_entities


caption_linked_entities = get_entities(caption_preprocessed_dir)
article_linked_entities = get_entities(article_preprocessed_dir)
