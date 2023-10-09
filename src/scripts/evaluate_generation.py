import json
import os
import sys
sys.path.append('.')
import re
from collections import defaultdict
from util.evaluate import _evaluate_news


def parse_generation(filename):
    f = open(filename, 'r')
    output = defaultdict(list)
    sent_id_set = set()
    current_key = None
    for line in f:
        m = re.match('^[STHDP]-[0-9]+[\t]', line)
        if m is not None:
            span = m.span()
            s = line[span[0] : span[1]]
            current_type = s[0]
            current_id = s[2:-1]
            sent_id_set.add(current_id)
            current_key = f'{current_type}-{current_id}'
            content = line[span[1]:]
            output[current_key].append(content)
        elif current_key is not None:
            output[current_key].append(line)

    annotations = {}
    results = {}
    for sent_id in sent_id_set:
        annotations[sent_id] = ' '.join(output[f'T-{sent_id}']).strip()
        result = ' '.join(output[f'D-{sent_id}'])       # e.g. result = "-0.9253601431846619\tMr. Hughes, left, and Mr. Hughes."
        match = re.match(r'^(-?\d+)(\.\d+)?\t', result)
        span = match.span()
        result_sent = result[span[1]:].strip()
        results[sent_id] = result_sent

    f.close()
    return annotations, results


def evaluate_fairseq_generation(output_path):
    annotations, results = parse_generation(output_path)
    print(len(annotations), len(results))
    metrics = _evaluate_news(annotations, results)
    print(metrics)
    return metrics


def evaluate_allennlp_generation(output_path):
    annotations = {}
    results = {}
    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            annotations[i] = obj['raw_caption']
            results[i] = obj['generation']
    print(len(annotations), len(results))
    metrics = _evaluate_news(annotations, results)
    print(metrics)
    return metrics


def main():
    # output_path = sys.argv[1]
    generation_path = sys.argv[1]

    # /media/wentian/sdb1/work/news_caption_fairseq/save/checkpoint_transformer_conv_wide_16head/test_results/generate-test.txt
    evaluate_fairseq_generation(generation_path)
    # evaluate_allennlp_generation("/media/wentian/sdb1/work/transform-and-tell-master/expt/goodnews/4_no_image/serialization/generations_40.jsonl")


if __name__ == '__main__':
    main()
