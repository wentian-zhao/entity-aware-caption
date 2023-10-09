import pickle

with open(r'/media/wentian/sdb1/work/news_caption/ner_goodnews_caption.pkl', 'rb') as f:
    d1 = pickle.load(f)

with open(r'/media/wentian/sdb1/work/news_caption/ner_nytimes_caption.pkl', 'rb') as f:
    d2 = pickle.load(f)

entity_types = set(key[1] for key in d1.keys())

with open(r'/root/image_crawler/entity_wiki_goodnews_with_images.txt', 'r') as f:
    names = [line.strip() for line in f.readlines()]

cnt_have_image = 0
d1_name = dict((key[0], key) for key in d1.keys())
for name in names:
    name = name.replace('_', ' ')
    if name in d1_name:
        cnt_have_image += 1