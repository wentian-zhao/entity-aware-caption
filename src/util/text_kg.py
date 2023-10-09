import json
from collections import OrderedDict
import numpy as np


def _span_to_tuple(span):
    return (span[0], (span[1][0], span[1][1]))


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


class TextGraph:
    def __init__(self):
        self.node_count = 0
        self.nodes_list = []
        self.node_to_index = OrderedDict()          # (0, (2, 4)) -> 0
        self.edges = list()

    def add_or_get_entity_id(self, span):
        span = _span_to_tuple(span)
        if span not in self.node_to_index:
            self.nodes_list.append(span)
            self.node_to_index[span] = self.node_count
            self.node_count += 1
        return self.node_to_index[span]

    def add_edge(self, u_span, v_span):
        u_span, v_span = _span_to_tuple(u_span), _span_to_tuple(v_span)
        u_id, v_id = self.node_to_index[u_span], self.node_to_index[v_span]
        if (u_id, v_id) not in self.edges:
            self.edges.append((u_id, v_id))

    def remove_nodes(self, node_index_list):
        new_g = TextGraph()
        for i in range(self.node_count):
            if i in node_index_list:
                continue
            span = self.nodes_list[i]
            new_g.add_or_get_entity_id(span)
        for (u, v) in self.edges:
            if u in node_index_list or v in node_index_list:
                continue
            u_span, v_span = self.nodes_list[u], self.nodes_list[v]
            new_g.add_edge(u_span, v_span)
        self.node_count = new_g.node_count
        self.nodes_list = new_g.nodes_list
        self.node_to_index = new_g.node_to_index
        self.edges = new_g.edges

def read_article_kg(d_article):
    sentences = d_article['sentences']
    entities = d_article['entities']
    corefs = d_article['corefs']

    coref_map_dict = dict()
    for key, value in corefs.items():
        ent_index_list = value

        for ent_index in ent_index_list:        # convert to the original span in the sentence
            sent_index, span = entities[ent_index]
            entities[ent_index] = [sent_index - 1, [span[0] - 1, span[1] - 1]]

        coref_chain_entities = [_get_words_from_article(d_article, entities[ent_index]) for ent_index in ent_index_list]
        # print('coref chain:', coref_chain_entities)

        for ent_index in ent_index_list:
            ent = entities[ent_index]
            ent_index_in_sent = entities.index(ent)
            mapped_ent_index = ent_index_list[0]
            ent = ent
            mapped_ent = entities[mapped_ent_index]
            # # print('\t', ent_index, ent, _get_words_from_token(sentences[ent[0]]['tokens'], ent[1]), '->', ent_index_sent, mapped_ent, _get_words_from_token(sentences[mapped_ent[0]]['tokens'], mapped_ent[1]))
            # print('\t', ent_index_sent, ent, _get_words_from_token(sentences[ent[0]]['tokens'], ent[1]), '->',
            #       mapped_ent_index, mapped_ent, _get_words_from_token(sentences[mapped_ent[0]]['tokens'], mapped_ent[1]))

            coref_map_dict[_span_to_tuple(ent)] = _span_to_tuple(mapped_ent)

    graph = TextGraph()
    for sent_index, sent in enumerate(d_article['sentences']):
        for triple in sent['triples']:
            subj_index, pred_index, obj_index, conf = triple
            subj_span, pred_span, obj_span = entities[subj_index], entities[pred_index], entities[obj_index]
            # subj_word, pred_word, obj_word = _get_words_from_token(tokens, subj_span[1]), _get_words_from_token(tokens, pred_span[1]), _get_words_from_token(tokens, obj_span[1])
            subj_span_mapped, obj_span_mapped = coref_map_dict.get(_span_to_tuple(subj_span), subj_span), coref_map_dict.get(_span_to_tuple(obj_span), obj_span)
            subj_word, pred_word, obj_word = _get_words_from_article(d_article, subj_span_mapped), \
                                             _get_words_from_article(d_article, pred_span), \
                                             _get_words_from_article(d_article, obj_span_mapped)

            # print('<{}, {}, {}>'.format(subj_word, pred_word, obj_word))

            graph.add_or_get_entity_id(subj_span_mapped)
            graph.add_or_get_entity_id(pred_span)
            graph.add_or_get_entity_id(obj_span_mapped)
            graph.add_edge(subj_span_mapped, pred_span)
            graph.add_edge(pred_span, obj_span_mapped)

    return graph
