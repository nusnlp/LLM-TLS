import json
import pickle
import os

from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
import operator
from evaluation import get_scores, evaluate_dates, get_average_results
from data import Dataset, get_average_summary_length
from datetime import datetime
from pprint import pprint
from collections import Counter
from tqdm import tqdm

from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.schema import Document
from keywords_mapping import TARGET_KEYWORDS

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import openai
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_average_summary_length(ref_tl):
    lens = [len(summary) for date, summary in ref_tl.dates_to_summaries.items()]
    return round(sum(lens) / len(lens))

def text_rank(sentences, embedding_func, personalization=None):
    sentence_embeddings = embedding_func.embed_documents(texts=sentences)
    cosine_sim_matrix = cosine_similarity(sentence_embeddings)
    nx_graph = nx.from_numpy_array(cosine_sim_matrix)
    scores = nx.pagerank(nx_graph, personalization=personalization)
    return sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

def get_pairs(event_pool):
    pairs, singletons = [], []
    for cluster_id, nodes in event_pool.items():
        if len(nodes) > 1:
            pairs.extend((min(nodes[i], nodes[j]), max(nodes[i], nodes[j])) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
        else:
            singletons.append(nodes[0])
    return pairs, singletons

def get_avg_score(scores):
    return sum(scores) / len(scores)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timelines_path", type=str, required=True)
    parser.add_argument("--events_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="./result")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--text_rank", action='store_true')
    parser.add_argument("--ds_path", type=str, default="./datasets/")
    return parser.parse_args()

evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
metric = 'align_date_content_costs_many_to_one'


def evaluate_timeline(overall_results, trials_path, collections, embedding_func, args):
    overall_trials = sorted(os.listdir(trials_path))
    # overall_results = {'rouge_1': [], 'rouge_2': [], 'date_f1': []}
    overall_r1_list, overall_r2_list, overall_d1_list = [], [], []

    for trial in overall_trials:
        print(f"Trial {trial}:")
        trial_res = process_trial(collections, trial, trials_path, args, embedding_func, overall_results)
        pprint(trial_res)
        
        rouge_1 = trial_res[0]['f_score']
        rouge_2 = trial_res[1]['f_score']
        date_f1 = trial_res[2]['f_score']
        trial_save_path = os.path.join(args.output, trial)
        os.makedirs(trial_save_path, exist_ok=True)

        overall_r1_list.append(rouge_1)
        overall_r2_list.append(rouge_2)
        overall_d1_list.append(date_f1)

        save_json(trial_res, os.path.join(trial_save_path, 'avg_score.json'))

    save_json(overall_results, os.path.join(args.output, 'global_result.json'))

    avg_r1 = get_avg_score(overall_r1_list)
    avg_r2 = get_avg_score(overall_r2_list)
    avg_d1 = get_avg_score(overall_d1_list)
    final_results = {
        'rouge1': avg_r1,
        'rouge2': avg_r2,
        'dateF1': avg_d1
    }
    print(final_results)

    save_json(final_results, os.path.join(args.output, 'average_result.json'))


def process_trial(collections, trial, trials_path, args, embedding_func, overall_results):
    results = []
    for keyword, index in tqdm(TARGET_KEYWORDS[args.dataset]):
    # for keyword, index in tqdm(collections):
        col = collections[index]
        events, content2type = load_events(os.path.join(args.events_path, f"{keyword.replace(' ', '_')}_events.jsonl"))
        for tl_index, gt_timeline in enumerate(col.timelines):
            summary_length = get_average_summary_length(TilseTimeline(gt_timeline.time_to_summaries))
            timeline_res = process_timeline(gt_timeline, summary_length, keyword, tl_index, trial, trials_path, args, embedding_func, content2type)
            (rouge_scores, date_scores, pred_timeline_dict) = timeline_res
            results.append(timeline_res)

            # Update overall results
            if keyword not in overall_results:
                overall_results[keyword] = {}
            if tl_index not in overall_results[keyword]:
                overall_results[keyword][tl_index] = []
            overall_results[keyword][tl_index].append((rouge_scores, date_scores))
    trial_res = get_average_results(results)
    return trial_res


def load_events(path):
    with open(path, 'r') as f:
        events = [json.loads(x) for x in f]
    content2type = {}
    for e in events:
        for key in ['llm', 'sentence']:
            if key in e:
                content = e[key].split(':')[-1].strip()
                content2type[content] = key
                break
    return events, content2type


def process_timeline(gt_timeline, summary_length, keyword, tl_index, trial, trials_path, args, embedding_func, content2type):
    trial_path = os.path.join(trials_path, trial, 'is_incremental', f"{keyword.replace(' ', '_')}/{tl_index}")
    docs = pickle.load(open(os.path.join(trial_path, 'docs.pickle'), 'rb'))
    event_pool = pickle.load(open(os.path.join(trial_path, 'event_pool.pickle'), 'rb'))

    times = gt_timeline.times
    L = len(times)

    # Generate pairs and clusters from event pool
    pairs, _ = get_pairs(event_pool)
    cluster_info = cluster_events(docs, pairs, L)

    pred_timeline = perform_text_summarization(cluster_info, summary_length, embedding_func, args.text_rank, content2type)

    # Evaluate summarization
    ground_truth = TilseGroundTruth([TilseTimeline(gt_timeline.date_to_summaries)])
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    rouge_scores = get_scores(metric, pred_timeline, ground_truth, evaluator)
    date_scores = evaluate_dates(pred_timeline, ground_truth)
    timeline_res = (rouge_scores, date_scores, pred_timeline)

    return timeline_res




def cluster_events(docs, pairs, top_l):
    # Build pools
    event_pool = dict()
    event2cluster = dict()
    for edge in pairs:
        event_id, event_near_id = edge[0], edge[1]
        cls_id = event2cluster.get(event_id, -1)
        neighbor_cls_id = event2cluster.get(event_near_id, -1)

        if cls_id == -1 and neighbor_cls_id == -1:
            new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
            event_pool[new_cls_id] = [event_id, event_near_id]
            event2cluster[event_id] = new_cls_id
            event2cluster[event_near_id] = new_cls_id
        elif cls_id == -1 and neighbor_cls_id != -1:
            event_pool[neighbor_cls_id] = event_pool[neighbor_cls_id] + [event_id]
            event2cluster[event_id] = neighbor_cls_id
        elif cls_id != -1 and neighbor_cls_id == -1:
            event_pool[cls_id] = event_pool[cls_id] + [event_near_id]
            event2cluster[event_near_id] = cls_id
        elif cls_id != -1 and neighbor_cls_id != -1 and cls_id != neighbor_cls_id:
            event_pool[cls_id] = list(set(event_pool[cls_id] + event_pool[neighbor_cls_id]))
            del event_pool[neighbor_cls_id]
            for e_id in event_pool[cls_id]:
                event2cluster[e_id] = cls_id
        else:
            pass
    
    total_ids = [doc['id'] for doc in docs]
    for node in total_ids:
        if node in event2cluster:
            continue 
        new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
        event_pool[new_cls_id] = [node]
        event2cluster[node] = new_cls_id
    
    
    clusters = list(event_pool.values())
    clusters.sort(key=len, reverse=True)

    docs_map = {}
    for doc in docs:
        id = doc['id']
        docs_map[id] = doc
    
    top_clusters = {}
    for idx, cluster in enumerate(clusters):
        # Get dates of events in this cluster
        dates = [docs_map[i]['timestamp'] for i in cluster]
        # Get the most often exist date
        cluster_date = max(set(dates), key=dates.count)

        if cluster_date not in top_clusters:
            top_clusters[cluster_date] = [list(cluster)]
        else:
            top_clusters[cluster_date].append(list(cluster))
        tmp = sorted(top_clusters[cluster_date], key=lambda x: len(x), reverse=True)
        top_clusters[cluster_date] = tmp
    
    for cluster_date in top_clusters.keys():
        tmp = []
        N = 1
        for i in range(N):
            tmp.extend(top_clusters[cluster_date][i])
        cnt = len(tmp)
        top_clusters[cluster_date] = (cnt, top_clusters[cluster_date][:N])
        # top_clusters[cluster_date] = (cnt, tmp)

    # print(f"within time range event cluster number: {keyword} {len(top_clusters)}")
    top_clusters = dict(sorted(top_clusters.items(), key=lambda item: item[1][0], reverse=True))
    top_clusters = list(top_clusters.values())


    # Assign core event
    cluster_info = []
    exist_dates = set()
    for cnt, same_day_clusters in top_clusters:
        all_events = []
        core_events = []
        cluster_date = None
        node_cnt = 0
        for cluster in same_day_clusters:
            node_cnt += len(cluster)
            # Get dates of events in this cluster
            dates = [docs_map[i]['timestamp'] for i in cluster]
            # Get the most often exist date
            cluster_date = max(set(dates), key=dates.count)
            # Get events which have the same date as the cluster date
            same_date_events = [docs_map[i] for i in cluster if docs_map[i]['timestamp'] == cluster_date]
            all_events.append([docs_map[i] for i in cluster])
            # Get core event (event with cluster date and cluster topic)
            core_event = next((event for event in same_date_events), None)
            core_events.append(core_event['text'].split(':')[-1].strip())
        # Prepare cluster info
        if cluster_date in exist_dates:
            continue
        exist_dates.add(cluster_date)
        cluster_info.append((cluster_date, node_cnt, core_events, all_events))
        if len(cluster_info) == (top_l):
            break

    # Sort clusters by date
    cluster_info.sort(key=operator.itemgetter(0))
    return cluster_info


def perform_text_summarization(cluster_info, summary_length, embedding_func, use_text_rank, content2type):
    pred_timeline_dict = {}
    # Print clusters
    for i, info in tqdm(enumerate(cluster_info)):

        date, num_events, core_events, cluster_events_lst = info
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        summary_list = []

        personalization = {}
        event_list = []
        for cluster_events in cluster_events_lst:
            for e in cluster_events:
                text = e['text'].split(':')[-1].strip()
                event_list.append(text)
                text_idx = len(event_list) - 1
                if content2type.get(text, "sentence") == 'llm':
                    personalization[text_idx] = 2.0
                else:
                    personalization[text_idx] = 1.0
                
        output = []
        if use_text_rank:
            ranked_list = [s for _, s in text_rank(event_list, embedding_func, personalization)]
            num_sentences = summary_length  # or whatever number of sentences you want in your summary
            output = [s for s in ranked_list[:num_sentences]]
        else:
            output = [s for s in event_list[:num_sentences]]
        summary_list.extend(output)

        pred_timeline_dict[date_obj] = summary_list
    pred_timeline_ = TilseTimeline(pred_timeline_dict)
    return pred_timeline_


def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    dataset_path = os.path.join(args.ds_path, args.dataset)
    dataset = Dataset(dataset_path)
    collections = dataset.collections
    embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

    evaluate_timeline({}, args.timelines_path, collections, embedding_func, args)


main()
