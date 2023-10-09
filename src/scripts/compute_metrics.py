import hashlib
import os
import pickle
import sys
import json
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc

cache_dir = '.evaluation_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

ann_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_mean_imggraph_goodnews/validation_results_test-small/annotation.json'
res_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_mean_imggraph_goodnews/validation_results_test-small/result_34_139253.json'


def spacize(text, cache, nlp):
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key not in cache:
        cache[key] = nlp(text).to_bytes()
    return Doc(nlp.vocab).from_bytes(cache[key])


def get_proper_nouns(doc):
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
    return proper_nouns


def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
        })
    return entities


# =========================


def compute_recall(obj):
    count = 0
    for name in obj['caption_names']:
        if name in obj['generated_names']:
            count += 1

    return count / len(obj['caption_names'])


def compute_precision(obj):
    count = 0
    for name in obj['generated_names']:
        if name in obj['caption_names']:
            count += 1

    return count / len(obj['generated_names'])


def compute_full_recall(obj):
    count = 0
    for name in obj['caption_names']:
        if name in obj['generated_names']:
            count += 1

    return count, len(obj['caption_names'])


def compute_full_precision(obj):
    count = 0
    for name in obj['generated_names']:
        if name in obj['caption_names']:
            count += 1

    return count, len(obj['generated_names'])


def compute_rare_recall(obj, counter):
    count = 0
    rare_names = [n for n in obj['caption_names'] if n not in counter]
    for name in rare_names:
        if name in obj['generated_names']:
            count += 1

    return count, len(rare_names)


def compute_rare_precision(obj, counter):
    count = 0
    rare_names = [n for n in obj['generated_names'] if n not in counter]
    for name in rare_names:
        if name in obj['caption_names']:
            count += 1

    return count, len(rare_names)


def contain_entity(entities, target):
    for ent in entities:
        # if ent['text'] == target['text'] and ent['label'] == target['label']:
        if ent['text'].lower() == target['text'].lower():
            return True
    return False


def compute_entities(obj, c):
    caption_entities = obj['caption_entities']
    gen_entities = obj['generated_entities']
    # context_entities = obj['context_entities']

    c['n_caption_ents'] += len(caption_entities)
    c['n_gen_ents'] += len(gen_entities)
    for ent in gen_entities:
        if contain_entity(caption_entities, ent):
            c['n_gen_ent_matches'] += 1
    for ent in caption_entities:
        if contain_entity(gen_entities, ent):
            c['n_caption_ent_matches'] += 1

    caption_persons = [e for e in caption_entities if e['label'] == 'PERSON']
    gen_persons = [e for e in gen_entities if e['label'] == 'PERSON']
    c['n_caption_persons'] += len(caption_persons)
    c['n_gen_persons'] += len(gen_persons)
    for ent in gen_persons:
        if contain_entity(caption_persons, ent):
            c['n_gen_person_matches'] += 1
    for ent in caption_persons:
        if contain_entity(gen_persons, ent):
            c['n_caption_person_matches'] += 1

    caption_orgs = [e for e in caption_entities if e['label'] == 'ORG']
    gen_orgs = [e for e in gen_entities if e['label'] == 'ORG']
    c['n_caption_orgs'] += len(caption_orgs)
    c['n_gen_orgs'] += len(gen_orgs)
    for ent in gen_orgs:
        if contain_entity(caption_orgs, ent):
            c['n_gen_orgs_matches'] += 1
    for ent in caption_orgs:
        if contain_entity(gen_orgs, ent):
            c['n_caption_orgs_matches'] += 1

    caption_gpes = [e for e in caption_entities if e['label'] == 'GPE']
    gen_gpes = [e for e in gen_entities if e['label'] == 'GPE']
    c['n_caption_gpes'] += len(caption_gpes)
    c['n_gen_gpes'] += len(gen_gpes)
    for ent in gen_gpes:
        if contain_entity(caption_gpes, ent):
            c['n_gen_gpes_matches'] += 1
    for ent in caption_gpes:
        if contain_entity(gen_gpes, ent):
            c['n_caption_gpes_matches'] += 1

    caption_date = [e for e in caption_entities if e['label'] == 'DATE']
    gen_date = [e for e in gen_entities if e['label'] == 'DATE']
    c['n_caption_date'] += len(caption_date)
    c['n_gen_date'] += len(gen_date)
    for ent in gen_date:
        if contain_entity(caption_date, ent):
            c['n_gen_date_matches'] += 1
    for ent in caption_date:
        if contain_entity(gen_date, ent):
            c['n_caption_date_matches'] += 1

    # TODO: evaluate rare named entities here

    return c


def compute_metrics(ann_file, res_file, cache_file, counters_file):
    nlp = spacy.load("en_core_web_lg")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    with open(ann_file, 'r') as f:
        d_ann = json.load(f)
    with open(res_file, 'r') as f:
        d_res = json.load(f)
    ann_dict = dict((item['image_id'], item['caption']) for item in d_ann['annotations'])
    res_dict = dict((item['image_id'], item['caption']) for item in d_res)

    with open(counters_file, 'rb') as f:
        counters = pickle.load(f)
    full_rp_counter = counters['context'] + counters['caption']

    recalls, precisions = [], []
    rare_recall, rare_recall_total = 0, 0
    rare_precision, rare_precision_total = 0, 0
    full_recall, full_recall_total = 0, 0
    full_precision, full_precision_total = 0, 0
    full_rare_recall, full_rare_recall_total = 0, 0
    full_rare_precision, full_rare_precision_total = 0, 0

    lengths, gt_lengths = [], []
    n_uniques, gt_n_uniques = [], []

    ent_counter = defaultdict(int)

    for image_id, caption in ann_dict.items():
        generation = res_dict[image_id]

        m_context = ''                 # ground-truth article from dataset
        # TODO: raw caption (from metadata) vs caption?
        m_caption = caption            # ground-truth caption from dataset

        caption_doc = spacize(m_caption, cache, nlp)
        gen_doc = nlp(generation)
        context_doc = spacize(m_context, cache, nlp)

        obj = {
            'caption': caption,
            'raw_caption': m_caption,
            'generation': generation,
            # 'copied_texts': copied_texts[i],
            # 'web_url': m['web_url'],
            # 'image_path': m['image_path'],
            'context': m_context,
            'caption_names': get_proper_nouns(caption_doc),
            'generated_names': get_proper_nouns(gen_doc),
            'context_names': get_proper_nouns(context_doc),
            'caption_entities': get_entities(caption_doc),
            'generated_entities': get_entities(gen_doc),
            'context_entities': get_entities(context_doc),
            # 'caption_readability': get_readability_scores(m['caption']),
            # 'gen_readability': get_readability_scores(generation),
            # 'caption_np': get_narrative_productivity(m['caption']),
            # 'gen_np': get_narrative_productivity(generation),
        }

        if obj['caption_names']:
            recalls.append(compute_recall(obj))
        if obj['generated_names']:
            precisions.append(compute_precision(obj))

        c, t = compute_full_recall(obj)
        full_recall += c
        full_recall_total += t

        c, t = compute_full_precision(obj)
        full_precision += c
        full_precision_total += t

        c, t = compute_rare_recall(obj, counters['caption'])
        rare_recall += c
        rare_recall_total += t

        c, t = compute_rare_precision(obj, counters['caption'])
        rare_precision += c
        rare_precision_total += t

        c, t = compute_rare_recall(obj, full_rp_counter)
        full_rare_recall += c
        full_rare_recall_total += t

        c, t = compute_rare_precision(obj, full_rp_counter)
        full_rare_precision += c
        full_rare_precision_total += t

        compute_entities(obj, ent_counter)

    metrics = {
        'All names - recall': {
            'count': full_recall,
            'total': full_recall_total,
            'percentage': (full_recall / full_recall_total) if full_recall_total else None,
        },
        'All names - precision': {
            'count': full_precision,
            'total': full_precision_total,
            'percentage': (full_precision / full_precision_total) if full_precision_total else None,
        },
        'Caption rare names - recall': {
            'count': rare_recall,
            'total': rare_recall_total,
            'percentage': (rare_recall / rare_recall_total) if rare_recall_total else None,
        },
        'Caption rare names - precision': {
            'count': rare_precision,
            'total': rare_precision_total,
            'percentage': (rare_precision / rare_precision_total) if rare_precision_total else None,
        },
        'Article rare names - recall': {
            'count': full_rare_recall,
            'total': full_rare_recall_total,
            'percentage': (full_rare_recall / full_rare_recall_total) if full_rare_recall_total else None,
        },
        'Article rare names - precision': {
            'count': full_rare_precision,
            'total': full_rare_precision_total,
            'percentage': (full_rare_precision / full_rare_precision_total) if full_rare_precision_total else None,
        },
        'Entity all - recall': {
            'count': ent_counter['n_caption_ent_matches'],
            'total': ent_counter['n_caption_ents'],
            'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
        },
        'Entity all - precision': {
            'count': ent_counter['n_gen_ent_matches'],
            'total': ent_counter['n_gen_ents'],
            'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
        },
        'Entity person - recall': {
            'count': ent_counter['n_caption_person_matches'],
            'total': ent_counter['n_caption_persons'],
            'percentage': ent_counter['n_caption_person_matches'] / ent_counter['n_caption_persons'],
        },
        'Entity person - precision': {
            'count': ent_counter['n_gen_person_matches'],
            'total': ent_counter['n_gen_persons'],
            'percentage': ent_counter['n_gen_person_matches'] / ent_counter['n_gen_persons'],
        },
        'Entity GPE - recall': {
            'count': ent_counter['n_caption_gpes_matches'],
            'total': ent_counter['n_caption_gpes'],
            'percentage': ent_counter['n_caption_gpes_matches'] / ent_counter['n_caption_gpes'],
        },
        'Entity GPE - precision': {
            'count': ent_counter['n_gen_gpes_matches'],
            'total': ent_counter['n_gen_gpes'],
            'percentage': ent_counter['n_gen_gpes_matches'] / ent_counter['n_gen_gpes'],
        },
        'Entity ORG - recall': {
            'count': ent_counter['n_caption_orgs_matches'],
            'total': ent_counter['n_caption_orgs'],
            'percentage': ent_counter['n_caption_orgs_matches'] / ent_counter['n_caption_orgs'],
        },
        'Entity ORG - precision': {
            'count': ent_counter['n_gen_orgs_matches'],
            'total': ent_counter['n_gen_orgs'],
            'percentage': ent_counter['n_gen_orgs_matches'] / ent_counter['n_gen_orgs'],
        },
        'Entity DATE - recall': {
            'count': ent_counter['n_caption_date_matches'],
            'total': ent_counter['n_caption_date'],
            'percentage': ent_counter['n_caption_date_matches'] / ent_counter['n_caption_date'],
        },
        'Entity DATE - precision': {
            'count': ent_counter['n_gen_date_matches'],
            'total': ent_counter['n_gen_date'],
            'percentage': ent_counter['n_gen_date_matches'] / ent_counter['n_gen_date'],
        },
    }

    return metrics


def main():
    cache_file = r'/media/wentian/sdb1/work/transform-and-tell-master/data/goodnews/evaluation_cache.pkl'
    counters_file = r'/media/wentian/sdb1/work/transform-and-tell-master/data/goodnews/name_counters.pkl'
    metrics = compute_metrics(ann_file, res_file, cache_file, counters_file)

    for key in ['All names', 'Caption rare names', 'Article rare names'] + ['Entity ' + category for category in ['all', 'person', 'GPE', 'ORG', 'DATE']]:
        p = metrics[key + ' - precision']['percentage']
        r = metrics[key + ' - recall']['percentage']
        f1 = 2 * p * r / (p + r)
        print('{:>25}\t{:>6f}\t{:>6f}\t{:>6f}'.format(key, p, r, f1))

    print(metrics)


if __name__ == '__main__':
    main()