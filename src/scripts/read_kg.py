import os
import sys
from collections import OrderedDict

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append('.')
import json
import config
from tqdm import tqdm
from util.text_kg import TextGraph, _span_to_tuple

from config import caption_processed_dirs, article_processed_dirs

p_dir = article_processed_dirs['goodnews']

files = os.listdir(p_dir)


def vis(d_article, graph):
    edge_with_labels = []
    for edge in graph.edges:
        u_span, v_span = graph.nodes_list[edge[0]], graph.nodes_list[edge[1]]
        u_word, v_word = _get_words_from_article(d_article, u_span), _get_words_from_article(d_article, v_span)
        # edge_with_labels.append((u_word, v_word))
        edge_with_labels.append((edge[0], edge[1]))

    G = nx.Graph()
    G.add_edges_from(edge_with_labels)
    nx.draw_networkx(G)
    plt.show()
    return None

def _get_words_from_token(tokens, span):
    t = tokens[span[0] : span[1]]
    return ' '.join([_[0] for _ in t])


def _get_words_from_article(d_article, span, type='words'):
    sent_index, word_span = span
    tokens = d_article['sentences'][sent_index]['tokens'][word_span[0] : word_span[1]]

    if type == 'words':
        return ' '.join(_[0] for _ in tokens)
    elif type == 'entity':
        ent_type_list = list(filter(lambda x: x is not None, (_[1] for _ in tokens if len(_) >= 2)))
        if len(ent_type_list) == 0:
            return None
        else:
            return ent_type_list[0]
    elif type == 'wiki':
        wiki_list = list(filter(lambda x: x is not None, (_[2] for _ in tokens if len(_) >= 3)))
        if len(wiki_list) == 0:
            return None
        else:
            return wiki_list[0]



if __name__ == '__main__':
    counter_node = Counter()
    counter_edge = Counter()
    counter_entity = Counter()

    for file in tqdm(files):
        file = os.path.join(p_dir, file)
        with open(file, 'r') as f:
            d = json.load(f)

        sentences = d['sentences']
        entities = d['entities']

        for sent_index, sent in enumerate(d['sentences']):
            tokens = sent['tokens']
            print('sent {}:'.format(sent_index), ' '.join(_[0] for _ in tokens))
            for triple in sent['triples']:
                subj_index, pred_index, obj_index, conf = triple
                subj, pred, obj = entities[subj_index], entities[pred_index], entities[obj_index]
                subj_span, pred_span, obj_span = _get_words_from_token(tokens, subj[1]), _get_words_from_token(tokens, pred[1]), _get_words_from_token(tokens, obj[1])
                print('\t{} - {} - {}'.format(subj, pred, obj))
                print('\t{} - {} - {}'.format(subj_span, pred_span, obj_span))

        # print('corefs:')
        # for key, value in d['corefs'].items():
        #     ent_index_list = value
        #     ent_words = []
        #     ent_spans = []
        #     for ent_index in ent_index_list:
        #         ent = entities[ent_index]
        #         sent_index, span = ent
        #         sent_index = sent_index - 1
        #         span = [span[0] - 1, span[1] - 1]
        #         ent = [sent_index, span]
        #
        #         # if sent_index >= len(sentences):
        #         #     continue
        #         words = _get_words_from_token(sentences[sent_index]['tokens'], span)
        #         ent_words.append(words)
        #         ent_spans.append(ent)
        #     print(ent_spans)
        #     print(ent_words)

        # ========

        coref_map_dict = dict()
        for key, value in d['corefs'].items():
            # print('corefs:', value)
            ent_index_list = value

            for ent_index in ent_index_list:        # to the original span in the sentence
                sent_index, span = entities[ent_index]
                entities[ent_index] = [sent_index - 1, [span[0] - 1, span[1] - 1]]

            for ent_index in ent_index_list:
                ent = entities[ent_index]
                ent_index_sent = entities.index(ent)
                mapped_ent_index = value[0]
                ent = ent
                mapped_ent = entities[mapped_ent_index]
                # # print('\t', ent_index, ent, _get_words_from_token(sentences[ent[0]]['tokens'], ent[1]), '->', ent_index_sent, mapped_ent, _get_words_from_token(sentences[mapped_ent[0]]['tokens'], mapped_ent[1]))
                # print('\t', ent_index_sent, ent, _get_words_from_token(sentences[ent[0]]['tokens'], ent[1]), '->',
                #       mapped_ent_index, mapped_ent, _get_words_from_token(sentences[mapped_ent[0]]['tokens'], mapped_ent[1]))

                coref_map_dict[_span_to_tuple(ent)] = _span_to_tuple(mapped_ent)

            mapped_ent_word = _get_words_from_article(d, entities[value[0]])
            mapped_ent_type = _get_words_from_article(d, entities[value[0]], type='entity')
            if mapped_ent_type is not None:
                counter_entity[(mapped_ent_word, mapped_ent_type)] += 1


        graph = TextGraph()
        for sent_index, sent in enumerate(d['sentences']):
            tokens = sent['tokens']
            # print('sent {}:'.format(sent_index), ' '.join(_[0] for _ in tokens))
            for triple in sent['triples']:
                subj_index, pred_index, obj_index, conf = triple
                subj_span, pred_span, obj_span = entities[subj_index], entities[pred_index], entities[obj_index]
                # subj_word, pred_word, obj_word = _get_words_from_token(tokens, subj_span[1]), _get_words_from_token(tokens, pred_span[1]), _get_words_from_token(tokens, obj_span[1])
                subj_span_mapped, obj_span_mapped = coref_map_dict.get(_span_to_tuple(subj_span), subj_span), coref_map_dict.get(_span_to_tuple(obj_span), obj_span)
                subj_word, pred_word, obj_word = _get_words_from_token(sentences[subj_span_mapped[0]]['tokens'], subj_span_mapped[1]), \
                                                 _get_words_from_token(sentences[pred_span[0]]['tokens'], pred_span[1]), \
                                                 _get_words_from_token(sentences[obj_span_mapped[0]]['tokens'], obj_span_mapped[1])

                # print('\t{} - {} - {}'.format(subj_word, pred_word, obj_word))
                # print('\t{} - {} - {}'.format(subj_span, pred_span, obj_span))
                # print('\t{} - {} - {}'.format(subj_span_mapped, pred_span, obj_span_mapped))

                graph.add_or_get_entity_id(subj_span_mapped)
                graph.add_or_get_entity_id(pred_span)
                graph.add_or_get_entity_id(obj_span_mapped)
                graph.add_edge(subj_span_mapped, pred_span)
                graph.add_edge(pred_span, obj_span_mapped)

        # vis(d, graph)
        counter_node[len(graph.node_to_index)] += 1
        counter_edge[len(graph.edges)] += 1

    print(counter_node)
    print(counter_edge)