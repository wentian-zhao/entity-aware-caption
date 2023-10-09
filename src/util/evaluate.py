import csv
import math
import os
import re
import types
import sys
import json
import traceback

import numpy as np

sys.path.append('.')

from config import coco_caption_path

try:
    sys.path.append(coco_caption_path)
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
except:
    traceback.print_exc()
    print('import coco-caption failed (from {})'.format(coco_caption_path))

try:
    from .customjson import dump_custom
    dump_func = dump_custom
except:
    dump_func = json.dump


class COCOResultGenerator:
    def __init__(self):
        self.result_obj = []
        self.annotation_obj = {'info': 'N/A', 'licenses': 'N/A', 'type': 'captions', 'images': [], 'annotations': []}
        self.caption_id = 0
        self.annotation_image_set = set()
        self.test_image_set = set()

    def add_annotation(self, image_id, caption_raw):
        # caption_raw = re.sub(r'[^\w\s]', '', caption_raw)
        caption_raw = re.sub(r'[\r|\n]+', ' ', caption_raw)

        if image_id not in self.annotation_image_set:
            self.annotation_obj['images'].append({'id': image_id})
            self.annotation_image_set.add(image_id)
        self.annotation_obj['annotations'].append({'image_id': image_id, 'caption': caption_raw, 'id': self.caption_id})
        self.caption_id += 1

    def add_output(self, image_id, caption_output, image_filename=None, metadata=None):
        # caption_output = re.sub(r'[^\w\s]', '', caption_output)
        caption_output = re.sub(r'[\r|\n]+', ' ', caption_output)

        assert(image_id in self.annotation_image_set and image_id not in self.test_image_set)
        item = {"image_id": image_id, "caption": caption_output}
        if metadata is not None:
            item['meta'] = metadata
        if image_filename is not None:
            item["image_filename"] = image_filename
        self.result_obj.append(item)
        self.test_image_set.add(image_id)

    def has_output(self, image_id):
        return image_id in self.test_image_set

    def get_annotation_and_output(self):
        return self.annotation_obj, self.result_obj

    def dump_annotation_and_output(self, annotation_file, result_file):
        self.dump_annotation(annotation_file)
        self.dump_output(result_file)

    def dump_annotation(self, annotation_file):
        with open(annotation_file, 'w') as f:
            print('dumping {} annotations to {}'.format(len(self.annotation_obj['annotations']), annotation_file))
            dump_func(self.annotation_obj, f, indent=4)

    def dump_output(self, result_file):
        with open(result_file, 'w') as f:
            print('dumping {} results to {}'.format(len(self.result_obj), result_file))
            dump_func(self.result_obj, f, indent=4)

    def add_img_scores(self, img_scores):
        """
        :param img_scores: [{'image_id': i, 'Bleu_1': 1, ...}, {'image_id': 0, 'Bleu_1': xx, }]
                returned by calling eval(ann_file, res_file, True)
        :return:
        """
        img_scores = dict([(i['image_id'], i) for i in img_scores])
        for item in self.result_obj:
            item.update(img_scores[item['image_id']])


def evaluate(ann_file, res_file, return_imgscores=False, use_scorers=('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr')):
    coco = COCO(ann_file)
    cocoRes = coco.loadRes(res_file)
    # create cocoEval object by taking coco and cocoRes
    # cocoEval = COCOEvalCap(coco, cocoRes)
    tokenizer = PTBTokenizer(lowercase=False)
    cocoEval = COCOEvalCap(coco, cocoRes, tokenizer=tokenizer, use_scorers=use_scorers)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    all_score = {}
    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        # print('%s: %.4f' % (metric, score))
        all_score[metric] = score

    img_scores = [cocoEval.imgToEval[key] for key in cocoEval.imgToEval.keys()]

    if return_imgscores:
        return all_score, img_scores
    else:
        return all_score



# Patch meteor scorer. See https://github.com/tylin/coco-caption/issues/25
def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()


def evaluate_news(ann_file, res_file, return_imgscores=False):
    with open(ann_file, 'r') as f:
        ann_obj = json.load(f)
    with open(res_file, 'r') as f:
        res_obj = json.load(f)

    annotations = {}
    results = {}

    for i in ann_obj['annotations']:
        # assert i['image_id'] not in annotations, 'duplicate image_id {} in annotation'.format(i['image_id'])
        if i['image_id'] in annotations:
            continue
        annotations[i['image_id']] = i['caption']
    for i in res_obj:
        # assert i['image_id'] not in results, 'duplicate image_id {} in result'.format(i['image_id'])
        if i['image_id'] in results:
            continue
        results[i['image_id']] = i['caption']
    assert annotations.keys() == results.keys()
    return _evaluate_news(annotations, results, return_imgscores)


def _evaluate_news(annotations, results, return_imgscores=False):
    # bleu, rouge, meteor, cider
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
    meteor_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    meteor_count = 0

    for image_id in annotations.keys():
        caption = annotations[image_id]     # ground truth
        generation = results[image_id]

        # remove punctuations
        caption = re.sub(r'[^\w\s]', '', caption)
        generation = re.sub(r'[^\w\s]', '', generation)

        caption = re.sub(r'[\r|\n]+', ' ', caption)
        generation = re.sub(r'[\r|\n]+', ' ', generation)

        caption = caption.lower()
        generation = generation.lower()

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])
        # meteor
        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        meteor_count += 1

    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(meteor_count):
        meteor_scores.append(float(
            meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    bleu_score, bleu_scores = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, cider_scores = cider_scorer.compute_score()

    metrics = {
        'Bleu_1': bleu_score[0],
        'Bleu_2': bleu_score[1],
        'Bleu_3': bleu_score[2],
        'Bleu_4': bleu_score[3],
        'METEOR': meteor_score,
        'ROUGE_L': rouge_score,
        'CIDEr': cider_score,
    }
    if return_imgscores:
        img_scores = {
            'Bleu_1': bleu_scores[0],
            'Bleu_2': bleu_scores[1],
            'Bleu_3': bleu_scores[2],
            'Bleu_4': bleu_scores[3],
            'METEOR': meteor_scores,
            'ROUGE_L': rouge_scores,
            'CIDEr': cider_scores,
        }
        return metrics, img_scores
    else:
        return metrics


def save_metrics(metric_file, metrics, epoch=None, global_step=None):
    lines = []
    if not os.path.exists(metric_file):
        first_line = ['epoch', 'step']
        for metric in metrics:
            first_line.append(metric)
        lines.append(','.join('{:<10}'.format(i) for i in first_line))
    else:
        with open(metric_file, 'r') as f:
            first_line = [i.strip() for i in f.readline().split(',')]
        if set(first_line[2:]) != set(metrics.keys()):
            print('existing metrics:', first_line[2:])
            print('received metrics:', list(metrics.keys()))

    strs = []
    for i in first_line:
        if i == 'epoch':
            strs.append('{:<10}'.format(epoch) if epoch else ' ' * 10)
        elif i == 'step':
            strs.append('{:<10}'.format(global_step) if global_step else ' ' * 10)
        else:
            strs.append('{:<10.6f}'.format(metrics[i]))
    lines.append(','.join(strs))
    with open(metric_file, 'a') as f:
        f.writelines([i + '\n' for i in lines])


if __name__ == '__main__':
    ann_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer2_conv_mm_image/validation_results_test-small/annotation.json'
    res_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer2_conv_mm_image/validation_results_test-small/result_50_204787.json'

    ann_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_nytimes/validation_results_test-small/annotation.json'
    # res_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_nytimes/validation_results_test-small/result_38_155640.json'
    res_file = r'/media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head_nytimes/validation_results_test-small/result_20_81916.json'

    metrics, imgscores = evaluate_news(ann_file, res_file, return_imgscores=True)

    with open(ann_file, 'r') as f:
        ann_obj = json.load(f)
    annotations = {}
    for i in ann_obj['annotations']:
        annotations[i['image_id']] = i['caption']

    keys = list(imgscores.keys())
    lines = [('image_id', 'gt', 'caption', *keys)]
    with open(res_file, 'r') as f:
        d = json.load(f)
        for i, item in enumerate(d):
            image_id = item['image_id']
            caption = item['caption']
            gt = annotations[image_id]
            scores = []
            for key in keys:
                scores.append(imgscores[key][i])
            lines.append((image_id, gt, caption, *scores))

    with open(res_file + '.csv', 'w', newline='', encoding='gb18030') as f:
        writer = csv.writer(f, )
        writer.writerows(lines)

    print(metrics)