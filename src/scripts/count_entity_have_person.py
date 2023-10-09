import os
import sys
import json

from tqdm import tqdm

dataset_name = 'goodnews'
dataset_name = 'nytimes'

preprocessed_dir = f'/media/wentian/nvme1/article_data/caption_preprocessed/{dataset_name}/'

files = os.listdir(preprocessed_dir)
image_have_person = set()
image_no_person = set()
for file in tqdm(files, ncols=80):
    f = open(os.path.join(preprocessed_dir, file), 'r')
    image_id = file.replace('.json', '')
    d = json.load(f)
    flag = False
    for sent in d['sentences']:
        for token in sent['tokens']:
            if len(token) >= 2 and token[1] == 'PERSON':
                flag = True
                break
    if flag: image_have_person.add(image_id)      # have person
    else: image_no_person.add(image_id)           # no person

cnt1, cnt2 = len(image_have_person), len(image_no_person)

with open(f'/media/wentian/sdb1/work/news_caption_fairseq/data/person_split_{dataset_name}.json', 'w') as f:
    json.dump({'have_person': list(image_have_person), 'no_person': list(image_no_person)}, f)

print('have person: {}, no person: {}, %(have person): {}'.format(cnt1, cnt2, cnt1 / (cnt1 + cnt2)))

