import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import operator
import os
import pickle
import random
import re
import time
import warnings
from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import openai
import pytz
import requests
from langchain.embeddings import (HuggingFaceEmbeddings,
                                  SentenceTransformerEmbeddings)
from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.evaluation import rouge
from tqdm import tqdm

from evaluation import (evaluate_dates, get_average_results, get_scores)
from keywords_mapping import TARGET_KEYWORDS
import utils
from data import Dataset


# use the IP or hostname of your instance
openai.api_key = "none"  # vLLM server is not authenticated

# model = SentenceTransformer('thenlper/gte-large')
embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

def convert_date_format(date_string):
    # Parse the date string into a datetime object
    parsed_date = datetime.strptime(date_string, '%Y-%m-%d')
    # Format the datetime object into the desired string format
    formatted_date = parsed_date.strftime('%B %d, %Y')
    # Remove leading zero of the day if present
    formatted_date = formatted_date.replace(' 0', ' ')
    return formatted_date


def completion_with_llm(prompt_str, temperature=0., max_len=512, stop_tokens=[], model="meta-llama/Llama-2-13b"):
    completion = openai.Completion.create(model=model,
                                        prompt=prompt_str,
                                        max_tokens=max_len,
                                        temperature=temperature,
                                        stop=stop_tokens,
                                      )
    # print("Completion result:", completion)
    return completion

def parsing_llm_output(llm_output):
    """
    Currently Only Using First Event Extraction
    """
    llm_output = llm_output.strip()
    pattern = r"(\d{4}-\d{2}-\d{2}): (.+)"
    matches = re.findall(pattern, llm_output)
    content, date = None, None
    contents, dates = [], []
    for idx, match in enumerate(matches):
        content = match[1]
        date = match[0]
        contents.append(content)
        dates.append(date)

    return contents, dates

def get_date_suffix(day):
    """
    Returns the suffix for the day of the month.
    """
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return suffix

def preprocess_events(events, keyword):
    """
    Preprocesses events based on the hyperparameter settings.
    Returns a list of Document objects.
    """
    docs = []
    event_des_dict = set()  # To keep track of unique event descriptions
    for idx, item in enumerate(events):
        title, date = item['title'], item['date']
        contents, dates = [], []
        if 'llm' in item:
            llm_output = item['llm']

            contents, dates = parsing_llm_output(llm_output)
            # if content is None or date is None:
            if len(contents) == 0 or len(dates) == 0:
                continue
        elif 'summary' in item:
            llm_output = item['summary']
            content = llm_output
            contents, dates = [content], [date]
        elif 'sentence' in item:
            llm_output = item['sentence']
            contents, dates = parsing_llm_output(llm_output)
            if len(contents) == 0 or len(dates) == 0:
                continue


        for content, date in zip(contents, dates):
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            except:
                continue
            
            text_content = content.strip()
            content = f'{str(date_obj)}: {text_content}'
            if content in event_des_dict:  # Don't add exactly same duplicate event description
                continue

            doc_index = len(docs)
            doc = Document(
                page_content=content,
                metadata={
                    'timestamp': str(date_obj),
                    'title': title,
                    'id': doc_index,
                    'keyword': keyword,
                    'content': text_content,
                }
            )
            docs.append(doc)
            event_des_dict.add(content)
    return docs # a list of event


def init_vector_db(docs, embedding_func):
    """
    Initializes the vector database with document embeddings.
    Returns the initialized database.
    """
    db = Chroma.from_documents(docs, embedding_func, collection_metadata={"hnsw:space": "cosine"})
    return db



def if_two_events_same(event1, event2, keyword, model, demonstrations, temperature=0., context=None,):
    """
    Determines if two events are the same.
    """
    prompt_str = f"{demonstrations}\n# Keyword\n{keyword}\n# Event 1\n{event1}\n# Event 2\n{event2}\n# Answer\n"

    completion, response = None, None

    completion = completion_with_llm(prompt_str, temperature, 2, stop_tokens=['\n----'], model=model)
    response = completion['choices'][0]['text']
    if random.random() < 0.05:
        print(response)
    if 'yes' in response.lower():
        return True, prompt_str
    return False, prompt_str


def generate_event_pool(docs, db, top_n, time_windows, kw, demonstrations, output_path, model, embedding_func=None, incremental=False):
    """
    Generates a pool of events. [Clustering]
    """

    event_pool = {}
    event2cluster = {}
    start_index = 0

    print("top_n: ", top_n)

    if not incremental:   # If not incremental clustering
        top_n = top_n + 1 # Retrieve top_n neighbors (+1 because the most similar one is the same as query string)
    else:
        assert embedding_func is not None
        try:
            db.delete_collection()
        except:
            pass
        db = Chroma.from_documents(docs[:1], embedding_func, collection_metadata={"hnsw:space": "cosine"})
        first_doc_id = docs[0].metadata['id']
        event_pool[-1] = [first_doc_id]
        event2cluster[first_doc_id] = -1
        start_index = 1
    
    pairwise_comparision_collector = []

    for item in tqdm(docs[start_index:], desc=kw):
        query = item.page_content
        query_title = item.metadata['title']
        # assert item.metadata['keyword'] == keyword

        timestamp = item.metadata['timestamp']
        
        TOPK = min(len(db.get()['ids']), top_n)

        event_neibors = db.similarity_search(query=query, k=TOPK)
        event_neibors = [d for d in event_neibors if d.page_content != query]
        
        
        has_same = False
        event_id = item.metadata['id']
        event_date = datetime.strptime(item.metadata['timestamp'], '%Y-%m-%d').date()

        # Remove out-of-timewindow events
        tmp = event_neibors.copy()
        event_neibors = []
        for doc in tmp:
            neighbor_date = datetime.strptime(doc.metadata['timestamp'], '%Y-%m-%d').date()
            date_diff = abs(event_date - neighbor_date) <= timedelta(days=time_windows)     # If not two events within the time window, skip.
            if not date_diff:
                continue
            event_neibors.append(doc)


        cls_id = -1

        for doc in event_neibors:
            cls_id = event2cluster.get(event_id, -1)

            event_near = doc.page_content
            # assert doc.metadata['keyword'] == keyword
            neighbor_date = datetime.strptime(doc.metadata['timestamp'], '%Y-%m-%d').date()

            event_near_id = doc.metadata['id']
            neighbor_cls_id = event2cluster.get(event_near_id, -1)
            neighbor_title = doc.metadata['title']

            if neighbor_cls_id == cls_id and neighbor_cls_id != -1:
                continue


            event1, event2 = query, event_near

            res, prompt_str = False, None

            context = None
            
            res, prompt_str = if_two_events_same(event1, event2, kw, model, temperature=0., context=context, demonstrations=demonstrations)
            
            if not res:
                continue
            
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
            has_same = True
            # break
        if not has_same and cls_id == -1:
            new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
            event_pool[new_cls_id] = [event_id]
            event2cluster[event_id] = new_cls_id
        
        if incremental:
            db.add_documents([item])
        
    # if incremental:   # Certain old Chroma DB does not support incremental inserting
    db.delete_collection()
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "event_pool.pickle"), 'wb') as handle:
        pickle.dump(event_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "event2cluster.pickle"), 'wb') as handle:
        pickle.dump(event2cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return event_pool, event2cluster


def preprocess_timeline_docs(docs, timeline):
    times = timeline.times
    start, end = times[0], times[-1]
    timeline_docs = []
    for doc in docs:
        doc_date = doc.metadata['timestamp']
        if datetime.strptime(doc_date, '%Y-%m-%d') < start:
            continue
        if datetime.strptime(doc_date, '%Y-%m-%d') > end:
            continue
        timeline_docs.append(doc)
    return timeline_docs


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b")
parser.add_argument("--keyword", type=str, nargs='*', required=True)
parser.add_argument("--output", type=str, default="./timeline_output")
parser.add_argument("--top_n", type=int, default=20)
parser.add_argument("--input", type=str, default="./event_outputs/entities/llama2-13b")
parser.add_argument("--incremental", action='store_true')
parser.add_argument("--time_windows", type=int, default=0)
parser.add_argument("--dataset", type=str, default='entities')
parser.add_argument("--port", type=int, default=-1)
parser.add_argument("--examples_path", type=str, default='./shots/entities/fewshot_0.json')
parser.add_argument("--ds_path", type=str, default="./datasets/")

if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model

    assert os.path.exists(args.input)

    time_windows = args.time_windows

    # Load dataset
    dataset_path = os.path.join(args.ds_path, args.dataset)
    dataset = Dataset(dataset_path)
    collections = dataset.collections

    title2context = {}

    print(f"{len(collections)} topics")

    print(f"Model: {args.model}")

    save_dir = None
    incremental_suffix = 'is_incremental' if args.incremental else 'not_incremental'
    openai.api_base = "http://0.0.0.0:8000/v1" 
    openai.api_key = "none"  # vLLM server is not authenticated
    trial = str(args.examples_path).split('_')[-1].split('.')[0]
    save_dir = os.path.join(args.output, f'{args.dataset}', args.model.split('/')[-1], trial, incremental_suffix)


    assert save_dir is not None
    os.makedirs(save_dir, exist_ok=True)
    

    
    demons_mapping = None
    with open(args.examples_path, 'r') as f:
        demons_mapping = json.load(fp=f)
        print(f"Using demonstrations from {args.examples_path}")
        print(f"Demonstrations has {len(demons_mapping)} keywords and {len(set(list(demons_mapping.values())))} types of demons.")

    # Acoording to vLLM, the API_BASE should set to local server with certain port
    if args.port != -1:
        openai.api_base = f"http://0.0.0.0:{args.port}/v1"

    print(f"OpenAI API_BASE: {openai.api_base}")
    
    results = []
    overall_start_time = time.time()
    for keyword, index in TARGET_KEYWORDS[args.dataset]:
        start_time = time.time()
        keyword_results = []
        if 'all' not in args.keyword:
            if keyword not in args.keyword:
                continue
        start_time = time.time()

        demons = demons_mapping[keyword]

        file_path = os.path.join(args.input, f"{keyword.replace(' ', '_')}_events.jsonl")
        assert os.path.exists(file_path), f'{file_path} not exist'
    
        output_path = os.path.join(save_dir, f"{keyword.replace(' ', '_')}")
        os.makedirs(output_path, exist_ok=True)

        col = collections[index]
        assert col.name == keyword.replace(' ', '_')

        kw = ' '.join(col.keywords)

        with open(file_path, "r") as f:
            data = [json.loads(x) for x in f]
        data = [item for item in data if any([i in item for i in ['llm', 'summary', 'sentence']])]
        docs = preprocess_events(data, keyword)

        print(f"{keyword}: {len(docs)} docs")


        for tl_idx, gt_timeline in enumerate(col.timelines):

            tl_output_path = os.path.join(output_path, str(tl_idx))
            os.makedirs(tl_output_path, exist_ok=True)

            times = gt_timeline.times
            L = len(times)
            
            tl_docs = preprocess_timeline_docs(docs, gt_timeline)

            db = None
            if not args.incremental:
                try:
                    db.dete_collection()
                except:
                    pass
                db = init_vector_db(tl_docs, embedding_func)
            
            tl_docs_dicts = []
            for d in tl_docs:
                tl_docs_dicts.append({
                    'text': d.page_content,
                    'timestamp': d.metadata['timestamp'],
                    'id': d.metadata['id']
                })
            with open(os.path.join(tl_output_path, 'docs.pickle'), 'wb') as handle:
                pickle.dump(tl_docs_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            event_pool, event2cluster = generate_event_pool(
                                                                docs=tl_docs,
                                                                db=db,
                                                                top_n=args.top_n,
                                                                time_windows=time_windows,
                                                                kw = keyword if args.dataset == 'entities' else kw,
                                                                demonstrations = demons,
                                                                output_path=tl_output_path,
                                                                model=model,
                                                                embedding_func=embedding_func,
                                                                incremental=args.incremental,
                                                            )
            
            clusters = list(event_pool.values())
            clusters.sort(key=len, reverse=True)

            top_clusters = {}
            for idx, cluster in enumerate(clusters):
                # Get dates of events in this cluster
                dates = [docs[i].metadata['timestamp'] for i in cluster]
                # Get the most often exist date
                cluster_date = max(set(dates), key=dates.count)

                if cluster_date not in top_clusters:
                    top_clusters[cluster_date] = list(cluster)
                else:
                    # top_clusters[cluster_date].extend(list(cluster))
                    if len(list(cluster)) > len(top_clusters[cluster_date]):
                        top_clusters[cluster_date] = cluster
            print(f"within time range event cluster number: {keyword} {len(top_clusters)}")
            top_clusters = dict(sorted(top_clusters.items(), key=lambda item: len(item[1]), reverse=True))
            top_clusters = list(top_clusters.values())


            # Assign core event
            cluster_info = []
            exist_dates = set()
            for cluster in top_clusters:
                # Get dates of events in this cluster
                dates = [docs[i].metadata['timestamp'] for i in cluster]
                # Get the most often exist date
                cluster_date = max(set(dates), key=dates.count)
                # Get events which have the same date as the cluster date
                same_date_events = [docs[i] for i in cluster if docs[i].metadata['timestamp'] == cluster_date]
                # Get core event (event with cluster date and cluster topic)
                core_event = next((event for event in same_date_events), None)
                # Prepare cluster info
                if cluster_date in exist_dates:
                    continue
                exist_dates.add(cluster_date)
                cluster_info.append((cluster_date, len(cluster), core_event.metadata['content']))

                # Probably need to remove
                if len(cluster_info) == (L):
                    break

            # Sort clusters by date
            cluster_info.sort(key=operator.itemgetter(0))

            pred_timeline_dict = {}
            # Print clusters
            for i, info in enumerate(cluster_info):
                date, num_events, core_event = info
                date_obj = datetime.strptime(date, '%Y-%m-%d').date()
                pred_timeline_dict[date_obj] = [core_event]

            evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
            metric = 'align_date_content_costs_many_to_one'

            pred_timeline_ = TilseTimeline(pred_timeline_dict)
            ground_truth = TilseGroundTruth([TilseTimeline(gt_timeline.date_to_summaries)])

            rouge_scores = get_scores(metric, pred_timeline_, ground_truth, evaluator)
            date_scores = evaluate_dates(pred_timeline_, ground_truth)
            duration = time.time() - start_time
            timeline_output = {
                'date_scores': date_scores,
                'rouge': rouge_scores,
                'pred': pred_timeline_dict,
                'target': gt_timeline.date_to_summaries,
                'time': duration,
            }
            score_obj = {
                'date_scores': date_scores,
                'rouge': rouge_scores,
                'time': duration,
            }

            with open(os.path.join(tl_output_path, f"timeline_output_{str(tl_idx)}.pickle"), 'wb') as handle:
                pickle.dump(timeline_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(tl_output_path, f"score_{str(tl_idx)}.json"), 'w') as f:
                json.dump(obj=score_obj, fp=f, indent=4)
            print(f'{keyword} Timeline {str(tl_idx)}, Alignment-based ROUGE:')
            pprint(rouge_scores)
            print(f'{keyword} Timeline {str(tl_idx)}, Date selection:')
            pprint(date_scores)
            print('-' * 100)
            results.append((rouge_scores, date_scores, pred_timeline_dict))
            keyword_results.append((rouge_scores, date_scores, pred_timeline_dict))
        avg_kw_results = get_average_results(keyword_results)
        duration = time.time() - start_time
        kw_result_obj = {
            'model': args.model,
            'top_n': args.top_n,
            'incremental': args.incremental,
            'average': avg_kw_results,
            'time': duration,
        }
        with open(os.path.join(output_path, 'avg_score.json'), 'w') as f:
            json.dump(fp=f, obj=kw_result_obj, indent=4)
    
    avg_results = get_average_results(results)
    print('Average results:')
    pprint(avg_results)
    all_duration = time.time() - overall_start_time
    result_obj = {
        'model': args.model,
        'top_n': args.top_n,
        'incremental': args.incremental,
        'average': avg_results,
        'time': all_duration,
    }
    with open(os.path.join(save_dir, 'avg_score.json'), 'w') as f:
        json.dump(fp=f, obj=result_obj, indent=4)

