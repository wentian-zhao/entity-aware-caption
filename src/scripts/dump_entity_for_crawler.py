import os
import json
import pickle
from collections import Counter, OrderedDict

from tqdm import tqdm

dataset_name = 'nytimes'

output_dir = '/media/wentian/sdb1/work/news_caption_fairseq/data/dumped_entities'

def dump_entity(dataset_name):
    data_dir = '/media/wentian/nvme1/article_data/preprocessed/{}'.format(dataset_name)
    c = Counter()

    for file in tqdm(os.listdir(data_dir)):
        with open(os.path.join(data_dir, file), 'r') as f:
            d = json.load(f)
        for sent in d['sentences']:
            for token in sent['tokens']:
                if len(token) == 3:
                    c[(token[1], token[2])] += 1

    print(len(c))

    with open(os.path.join(output_dir, 'entity_wiki_{}_all.pkl'.format(dataset_name)), 'wb') as f:
        pickle.dump(c, f)

    others_entity_types = {'CITY', 'COUNTRY', 'LOCATION', 'ORGANIZATION', 'STATE_OR_PROVINCE'}

    persons_dict = dict()
    others_dict = dict()
    for (entity_type, entity_name), count in c.items():
        if entity_type == 'PERSON':
            persons_dict[len(persons_dict)] = (entity_type, entity_name, count)
        elif entity_type in others_entity_types:
            others_dict[len(others_dict)] = (entity_type, entity_name, count)

    with open(os.path.join(output_dir, 'entity_wiki_{}_persons.pkl'.format(dataset_name)), 'wb') as f:
        pickle.dump(persons_dict, f)
    with open(os.path.join(output_dir, 'entity_wiki_{}_others.pkl'.format(dataset_name)), 'wb') as f:
        pickle.dump(others_dict, f)

# # 'PERSON', 'COUNTRY', 'STATE_OR_PROVINCE', 'CITY', 'ORGANIZATION', 'LOCATION',

# 'CITY', 'COUNTRY', 'LOCATION', 'ORGANIZATION', 'PERSON', 'STATE_OR_PROVINCE',


if __name__ == '__main__':
    dump_entity('goodnews')
    dump_entity('nytimes')