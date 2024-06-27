import warnings
warnings.filterwarnings(action='ignore')

from collections import OrderedDict
from datetime import datetime, timedelta

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from itertools import chain
import json
import os

import re
from tqdm import tqdm

from datasets import Dataset, load_metric, load_dataset
import nltk
import numpy as np
import openai
from peft import PeftModel
import tiktoken
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM , AutoTokenizer, DataCollatorWithPadding
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import argparse
from datasets import Dataset, load_metric, load_dataset
metric = load_metric("rouge")

embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')



def clean_tweet(tweet):
    """Removes URLs from the tweet"""
    c_tweet = re.sub(r'http\S+', '', tweet).strip()
    c_tweet = re.sub('\s+', ' ', c_tweet).strip()
    return c_tweet

def read_data(path):
    try:
        with open(path, mode='r') as f:
            data = json.loads(f.read())
    except:
        with open(path, mode='r') as f:
            data = [json.loads(x) for x in f]
    return data

def parse_timestamp(timestamp):
    if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2}', timestamp):
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S%z')
    elif re.match(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}', timestamp):
        return datetime.strptime(timestamp, '%m/%d/%Y, %H:%M:%S')
    elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', timestamp):
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    else:
        raise ValueError(f'Unknown timestamp format: {timestamp}')

def postprocess_text(preds, labels):
    preds = [pred.strip().strip('</s>') for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def find_temporal_center(timestamps):
    if isinstance(timestamps[0], datetime):
        parsed_timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    else:
        parsed_timestamps = [parse_timestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    timestamps_in_seconds = [datetime.strptime(parsed_ts, "%Y-%m-%d %H:%M:%S").timestamp() for parsed_ts in parsed_timestamps]
    temporal_center = datetime.fromtimestamp(np.mean(timestamps_in_seconds))
    return temporal_center


def get_global_tweets(data):
    global_tweets = []
    for timeline in data:
        timeline_id = timeline['batch_id'] + '_' + timeline['timeline_id']
        tweet_ids = timeline['tweet_ids']
        tweets = timeline['tweets']
        times = timeline['times']
        labels = ["1"] + timeline['labels']
        majority_reasons = ["NA"] + timeline['majority_reasons']

        for tweet_id, tweet, time, label, majority_reason in zip(tweet_ids, tweets, times, labels, majority_reasons):
            timestamp = parse_timestamp(time).strftime('%Y-%m-%d %H:%M:%S')
            tweet_str = timestamp + ": " + clean_tweet(tweet)
            # if label == '1':
            global_tweets.append({
                'id': tweet_id,
                'timeline_id': timeline_id,
                'text': tweet_str,
                'timestamp': timestamp,
                'label': label,
                'majority_reason': majority_reason,
            })

    # Tweets sort by time
    global_tweets.sort(key=lambda x: x['timestamp'])
    print(f'Total # of Tweets : {len(global_tweets)}')
    return global_tweets


openai.api_key = "none"
openai.api_base = "http://0.0.0.0:8000/v1"


def completion_with_llm(prompt_str, temperature=0., max_len=512, stop_tokens=[], model_name=None):
    completion = openai.Completion.create(
        model=model_name,
        prompt=prompt_str,
        max_tokens=max_len,
        temperature=temperature,
        stop=stop_tokens,
    )
    return completion

def binary_classification(prompt_str, model_name):
    completion = completion_with_llm(prompt_str, temperature=0., max_len=10, stop_tokens='----', model_name=model_name)
    output = completion.choices[0].text
    return output

def tweet_to_doc(twt):
    doc = Document(page_content=twt['text'], metadata= {
                'id': twt['id'],
                'timeline_id': twt['timeline_id'],
                'text': twt['text'],
                'timestamp': twt['timestamp'],
                'label': twt['label'],
                'majority_reason': twt['majority_reason']
            })
    return doc

def postprocess_text(preds, labels):
    preds = [pred.strip().strip('</s>') for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    try:
        decoded_preds = tokenizer.decode_batch(preds)
        decoded_labels = tokenizer.decode_batch(labels)
    except:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def rouge_eval(path, tokenizer):
    # tokenizer = tiktoken.encoding_for_model("gpt2")
    # tokenizer = AutoTokenizer.from_pretrained("/home/llm2/models/llama2/llama-2-13b-hf")
    with open(path, 'r') as f:
        results = [json.loads(x) for x in f.readlines()]

    preds = [item['output'] for item in results]
    gold = [item['references'] for item in results]

    new_preds = []
    new_gold = []
    for p, g in zip(preds, gold):
        new_preds.append(p)
        new_preds.append(p)
        new_gold.append(g[0])
        new_gold.append(g[1])

    try:
        eval_preds = tokenizer.encode_batch(new_preds), tokenizer.encode_batch(new_gold)
    except:
        eval_preds = tokenizer(new_preds)['input_ids'], tokenizer(new_gold)['input_ids']
    result = compute_metrics(eval_preds, tokenizer)

    return result

BINARY_CLASSIFICATION_PROMPT = """You are given a list of tweets, in chronological order, about some event in a timeline below:
 
{timeline_tweets}

Consider the following new tweet:

{new_tweet}

If this new tweet follows the given list of tweets in the same timeline (i.e., the new tweet is about the same event), then reply with “Yes”.
 
If the new tweet is not relevant to the event in the given timeline, then reply with “No, it is not relevant”.
 
If the new tweet is repetitive or redundant (i.e., it repeats information present in previous tweets in the given timeline), then reply with “No, it is repetitive”.
 
If a new tweet is not informative (i.e., it is generic or expresses some opinion but does not add new information to the event in the given timeline), then reply with “No, it is not informative”.

# Answer
"""


def retrieval_cluster(global_tweets, cls_model_name):
    RETRIEVAL = True
    clusters = []
    first_tweet = global_tweets[0]
    id2cluster = {}
    tweet2clusterid = {}
    id2cluster[-1] = []
    id2cluster[-2] = []

    top_n = 20

    try:
        db.delete_collection()
    except:
        pass
    if RETRIEVAL:
        first_doc = tweet_to_doc(first_tweet)
        db = Chroma.from_documents([first_doc], embedding_func, collection_metadata={"hnsw:space": "cosine"})

    clusters.append([first_tweet]) # intialize the cluster
    id2cluster[0] = clusters[0]
    tweet2clusterid[first_tweet['id']] = 0

    rep_clusters = []
    non_inf_clusters = []

    def dedup_retr_clusters(retr_tweets):
        clusters_ids = set()
        for twt in retr_tweets:
            id = twt.metadata['id']
            cls_id = tweet2clusterid[id]
            if cls_id < 0:
                continue
            clusters_ids.add(cls_id)
        return list(clusters_ids)


    for tweet in tqdm(global_tweets[1:]): # incoming stream of tweets
        doc = tweet_to_doc(tweet)
        pred_clusters_idx = []
        is_rep = False
        is_not_inf = False
        tweet_id = doc.metadata['id']

        TOPK = min(len(db.get()['ids']), top_n)
        query = doc.page_content
        retr_tweets = db.similarity_search(query=query, k=TOPK)
        
        target_clusters_ids = dedup_retr_clusters(retr_tweets)
        
        for cls_id in target_clusters_ids:
            cluster = id2cluster[cls_id]
            cluster_tweets = [twt['text'] for twt in cluster]
            tweet_timestamp = tweet['timestamp']
            latest_tweet = cluster[-1]
            time_diff = abs(datetime.strptime(tweet_timestamp, '%Y-%m-%d %H:%M:%S') - datetime.strptime(latest_tweet['timestamp'], '%Y-%m-%d %H:%M:%S'))
            if time_diff > timedelta(days=1): # use date heuristics to improve clustering (only cluster when within 1 day(24 hours) from the latest tweet of the cluster)
                continue
            prompt_str = BINARY_CLASSIFICATION_PROMPT.format(
                timeline_tweets='\n'.join(cluster_tweets),
                new_tweet=tweet['text'],
            )
            output = binary_classification(prompt_str, cls_model_name) # binary_classification() returns the response from LLM
            if 'yes' in output.lower():
                pred_clusters_idx.append(cls_id)
            else:
                if 'repetitive' in output.lower(): # if repetitive, skip the tweet and not create a new cluster
                    is_rep = True
                    break
                if 'informative' in output.lower(): # if informative, skip the tweet and not create a new cluster
                    is_not_inf = True
                    break
        
        if is_rep:
            rep_clusters.append(tweet)
            id2cluster[-1].append(tweet)
            tweet2clusterid[tweet_id] = -1
            continue
        if is_not_inf:
            non_inf_clusters.append(tweet)
            id2cluster[-2].append(tweet)
            tweet2clusterid[tweet_id] = -2
            continue
        
        db.add_documents([doc])
        num_pred_clusters = len(pred_clusters_idx)
        if num_pred_clusters == 1:
            # 1) only one pred cluster -> merge to the pred cluster
            clusters[pred_clusters_idx[0]].append(tweet)
            id2cluster[pred_clusters_idx[0]].append(tweet)
            tweet2clusterid[tweet_id] = pred_clusters_idx[0]
            
        elif num_pred_clusters > 1:
            # 2) more than one pred clusters -> merge to the pred cluster with the nearest time
            tweet_timestamp = tweet['timestamp']
            min_idx = -1
            min_diff = timedelta.max
            for pred_cluster_idx in pred_clusters_idx:
                latest_tweet = id2cluster[pred_cluster_idx][-1]
                time_diff = abs(datetime.strptime(tweet_timestamp, '%Y-%m-%d %H:%M:%S') - datetime.strptime(latest_tweet['timestamp'], '%Y-%m-%d %H:%M:%S'))
                if time_diff < min_diff:
                    min_idx = pred_cluster_idx
                    min_diff = time_diff
            clusters[min_idx].append(tweet)
            id2cluster[min_idx].append(tweet)
            tweet2clusterid[tweet_id] = min_idx
            
        else:
            # 3) no pred cluster -> create new cluster
            new_cls_id = max(list(id2cluster.keys())) + 1
            clusters.append([tweet])
            id2cluster[new_cls_id] = [tweet]
            # id2cluster[new_cls_id].append(tweet)
            tweet2clusterid[tweet_id] = new_cls_id
    
    return clusters


def global_cluster(global_tweets, cls_model_name):
    RETRIEVAL = False
    clusters = []
    first_tweet = global_tweets[0]

    clusters.append([first_tweet]) # intialize the cluster

    rep_clusters = []
    non_inf_clusters = []

    for tweet in tqdm(global_tweets[1:]): # incoming stream of tweets
        pred_clusters_idx = []
        is_rep = False
        is_not_inf = False
        for c_idx, cluster in enumerate(clusters):
            cluster_tweets = [timeline['text'] for timeline in cluster]
            tweet_timestamp = tweet['timestamp']
            latest_tweet = clusters[c_idx][-1]
            time_diff = abs(datetime.strptime(tweet_timestamp, '%Y-%m-%d %H:%M:%S') - datetime.strptime(latest_tweet['timestamp'], '%Y-%m-%d %H:%M:%S'))
            if time_diff > timedelta(days=1): # use date heuristics to improve clustering (only cluster when within 1 day(24 hours) from the latest tweet of the cluster)
                # print('NO (date)')
                continue
            prompt_str = BINARY_CLASSIFICATION_PROMPT.format(
                timeline_tweets='\n'.join(cluster_tweets),
                new_tweet=tweet['text'],
            )
            output = binary_classification(prompt_str, cls_model_name) # binary_classification() returns the response from LLM
            if 'yes' in output.lower():
                pred_clusters_idx.append(c_idx)
                # print('YES')
            else:
                if 'repetitive' in output.lower(): # if repetitive, skip the tweet and not create a new cluster
                    is_rep = True
                    # print('NO (repetitive)')
                    break
                if 'informative' in output.lower(): # if informative, skip the tweet and not create a new cluster
                    is_not_inf = True
                    # print('NO (informative)')
                    break
                # print('NO')
        
        if is_rep:
            rep_clusters.append(tweet)
            continue
        if is_not_inf:
            non_inf_clusters.append(tweet)
            continue
        num_pred_clusters = len(pred_clusters_idx)
        if num_pred_clusters == 1:
            # 1) only one pred cluster -> merge to the pred cluster
            clusters[pred_clusters_idx[0]].append(tweet)
        elif num_pred_clusters > 1:
            # 2) more than one pred clusters -> merge to the pred cluster with the nearest time
            tweet_timestamp = tweet['timestamp']
            min_idx = -1
            min_diff = timedelta.max
            for pred_cluster_idx in pred_clusters_idx:
                latest_tweet = clusters[pred_cluster_idx][-1]
                time_diff = abs(datetime.strptime(tweet_timestamp, '%Y-%m-%d %H:%M:%S') - datetime.strptime(latest_tweet['timestamp'], '%Y-%m-%d %H:%M:%S'))
                if time_diff < min_diff:
                    min_idx = pred_cluster_idx
                    min_diff = time_diff
            clusters[min_idx].append(tweet)
        else:
            # 3) no pred cluster -> create new cluster
            clusters.append([tweet])
    return clusters


def save_cluster_info(clusters, path, split='test'):
    num_tweet = 0
    txt_file_name = os.path.join(path, f'cluster_result_{split}.txt')
    json_file_name = os.path.join(path, f'cluster_{split}.json')
    with open(txt_file_name, 'w') as f:
        for idx, cluster in enumerate(clusters):
            f.write(f'=====CLUSTER #{idx+1}====\n')
            for tweet in cluster:
                f.write(f"[{tweet['timeline_id']}({tweet['label']}:{tweet['majority_reason']})] {tweet['text']}\n")
                num_tweet += 1
        f.write(f'\nTotal # of Clusters : {len(clusters)}')
        f.write(f'\nTotal # of Tweets : {num_tweet}')

    with open(json_file_name, 'w') as f:
        json.dump(clusters, f)

def get_gold_timelines(data):
    gold_timelines = []
    for timeline in data:
        timeline_id = timeline['batch_id'] + '_' + timeline['timeline_id']
        tweet_ids = timeline['tweet_ids']
        tweets = timeline['tweets']
        times = timeline['times']
        labels = ["1"] + timeline['labels']
        majority_reasons = ["NA"] + timeline['majority_reasons']

        gold_tweet_ids = []
        gold_texts = []
        gold_timestamps = []

        for tweet_id, tweet, time, label, majority_reason in zip(tweet_ids, tweets, times, labels, majority_reasons):
            timestamp = parse_timestamp(time).strftime('%Y-%m-%d %H:%M:%S')
            tweet_str = timestamp + ": " + clean_tweet(tweet)
            if label == '1':
                gold_tweet_ids.append(tweet_id)
                gold_texts.append(tweet_str)
                gold_timestamps.append(timestamp)
            
        gold_timelines.append({
            'timeline_id': timeline_id,
            'gold_tweet_ids': gold_tweet_ids,
            'gold_texts': gold_texts,
            'gold_timestamps': gold_timestamps,
        })

    # Timelines sort by time (of seed tweet)
    gold_timelines.sort(key=lambda x: x['gold_timestamps'][0])
    print(f'Total # of Gold Timelines : {len(gold_timelines)}')
    return gold_timelines

def get_gold_pred_mapping(gold_timelines, clusters):
    gold2pred_mapping = {}
    matched_pred_clusters = []
    no_match_gold_timeline_ids = []

    for gold_timeline in tqdm(gold_timelines):
        matched_pred_cluster_id = -1
        max_overlapping_tweets_num = 0
        matched_cluster_len = -1
        gold_timeline_id = gold_timeline['timeline_id']
        gold_timeline_tweet_ids = gold_timeline['gold_tweet_ids']
        gold_timeline_tweets = gold_timeline['gold_texts']
        gold_timeline_timestamps = gold_timeline['gold_timestamps']

        for c_idx, cluster in enumerate(clusters):
            if c_idx in matched_pred_clusters: # ensure that a prediction cluster does not align with more than one gold cluster
                continue
            cluster_tweet_ids = [c['id'] for c in cluster]
            overlapping_tweets_num = len(set(gold_timeline_tweet_ids) & set(cluster_tweet_ids)) # check for overlapping tweets between gold timeline and cluster using tweet ids
            is_match = False
            if overlapping_tweets_num > 0: # ensure that the alignment pair has at least more than 0 overlapping tweets
                if overlapping_tweets_num > max_overlapping_tweets_num:
                    is_match = True
                elif overlapping_tweets_num == max_overlapping_tweets_num: # tie-breaker (choose the prediction cluster with a higher precision score i.e., less number of tweets)
                    print(f'!!!Tie between Matched Cluster #{matched_pred_cluster_id+1} and Cluster #{c_idx+1} for Overlapping Tweets of {overlapping_tweets_num}!!!')
                    if len(cluster_tweet_ids) < matched_cluster_len:
                        is_match = True
                    elif len(cluster_tweet_ids) == matched_cluster_len: # tie-breaker (choose the prediction cluster with a closer temporal center to that of the gold cluster)
                        print(f'!!!Tie between Matched Cluster #{matched_pred_cluster_id+1} and Cluster #{c_idx+1} for Cluster Size of {len(cluster_tweet_ids)}!!!')
                        # compare temporal centers
                        cluster_temporal_center = find_temporal_center([c['timestamp'] for c in cluster])
                        matched_cluster_temporal_center = find_temporal_center([c['timestamp'] for c in clusters[matched_pred_cluster_id]])
                        gold_temporal_center = find_temporal_center(gold_timeline_timestamps)
                        time_diff = abs(cluster_temporal_center - gold_temporal_center)
                        matched_time_diff = abs(matched_cluster_temporal_center - gold_temporal_center)
                        if time_diff < matched_time_diff:
                            is_match = True
            if is_match:
                max_overlapping_tweets_num = overlapping_tweets_num
                matched_cluster_len = len(cluster_tweet_ids)
                matched_pred_cluster_id = c_idx

        if matched_pred_cluster_id == -1:  # ensure that the alignment pair has at least more than 0 overlapping tweet
            print(f"!!!Timeline {gold_timeline_id} not mapped with any clusters!!!")
            gold2pred_mapping[gold_timeline_id] = matched_pred_cluster_id # align gold timelines with no matching clusters with cluster #-1
            matched_pred_clusters.append(matched_pred_cluster_id)
            no_match_gold_timeline_ids.append(gold_timeline_id)
        else:    
            print(f'Overlapping Tweets for Timeline {gold_timeline_id} and Cluster #{matched_pred_cluster_id+1} :')
            for overlap_tweet in (set(gold_timeline_tweets) & set([i['text'] for i in clusters[matched_pred_cluster_id]])):
                print(f'\t{overlap_tweet}')
            
            # match_id, match_num, len(timestamp_tweets)
            gold2pred_mapping[gold_timeline_id] = matched_pred_cluster_id
            matched_pred_clusters.append(matched_pred_cluster_id)
            print(f"Timeline {gold_timeline_id} matched with Cluster #{matched_pred_cluster_id+1}.")
        print('='*100)
    return gold2pred_mapping

def cluster_evaluate(gold_timelines, clusters, output_path, DATASET_TYPE):
    gold2pred_mapping = get_gold_pred_mapping(gold_timelines, clusters)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    file_name = os.path.join(output_path, f'cluster_alignment_result_{DATASET_TYPE}.txt')
    with open(file_name, 'w') as f:
        for gold_timeline in tqdm(gold_timelines):
            matched_pred_cluster_id = -1
            max_overlapping_tweets_num = 0
            matched_cluster_len = -1
            gold_timeline_id = gold_timeline['timeline_id']
            gold_timeline_tweet_ids = gold_timeline['gold_tweet_ids']
            gold_timeline_tweets = gold_timeline['gold_texts']
            gold_timeline_timestamps = gold_timeline['gold_timestamps']

            matched_cluster_id = gold2pred_mapping.get(gold_timeline_id)
            if matched_cluster_id == -1: # gold cluster did not match with any prediction cluster
                f.write(f'==========TIMELINE #{gold_timeline_id}==========\n')
                for g_tweet_id, g_tweet in zip(gold_timeline_tweet_ids, gold_timeline_tweets):
                    g_tweet_str = g_tweet
                    f.write(f'{g_tweet_str}\n')
                f.write(f'==========RESULT==========\n')
                f.write(f'No. of Overlapping Tweets : 0\n')
                f.write(f'No. of Tweets in Timeline : {len(gold_timeline_tweet_ids)}\n')
                f.write(f'No. of Tweets in Matched Cluster : N/A\n')
                f.write('\n')

                accuracy_scores.append(0) # TP + TN = Total - (FP + FN)
                precision_scores.append(0)
                recall_scores.append(0)
            else:
                matched_cluster = clusters[matched_cluster_id]
                cluster_tweet_ids = [tweet['id'] for tweet in matched_cluster]
                cluster_tweets = [tweet['text'] for tweet in matched_cluster]

                overlap_tweet_ids = (set(gold_timeline_tweet_ids) & set(cluster_tweet_ids))

                f.write(f'==========TIMELINE #{gold_timeline_id}==========\n')
                for g_tweet_id, g_tweet in zip(gold_timeline_tweet_ids, gold_timeline_tweets):
                    g_tweet_str = g_tweet
                    if g_tweet_id in overlap_tweet_ids:
                        f.write(f'[O] {g_tweet_str}\n')
                    else:
                        f.write(f'{g_tweet_str}\n')

                f.write(f'==========CLUSTER #{matched_cluster_id+1}==========\n')
                for c_tweet_id, c_tweet in zip(cluster_tweet_ids, cluster_tweets):
                    c_tweet_str = c_tweet
                    if c_tweet_id in overlap_tweet_ids:
                        f.write(f'[O] {c_tweet_str}\n')
                    else:
                        f.write(f'{c_tweet_str}\n')

                f.write(f'==========RESULT==========\n')
                f.write(f'No. of Overlapping Tweets : {len(overlap_tweet_ids)}\n')
                f.write(f'No. of Tweets in Timeline : {len(gold_timeline_tweet_ids)}\n')
                f.write(f'No. of Tweets in Matched Cluster : {len(cluster_tweet_ids)}\n')
                f.write('\n')

                accuracy_scores.append((len(gold_timelines)-((len(cluster_tweets)-len(overlap_tweet_ids))+(len(gold_timeline_tweet_ids)-len(overlap_tweet_ids))))/len(gold_timelines)*1.0) # TP + TN = Total - (FP + FN)
                precision_scores.append(len(overlap_tweet_ids)/len(cluster_tweet_ids)*1.0)
                recall_scores.append(len(overlap_tweet_ids)/len(gold_timeline_tweet_ids)*1.0)

    for p_score, r_score in zip(precision_scores, recall_scores):
        if p_score == 0 and r_score == 0:
            f1_score = 0
        else:
            f1_score = (2.0 * (p_score * r_score)) / (p_score + r_score)
        f1_scores.append(f1_score)
    
    avg_prec = sum(precision_scores)/len(precision_scores)
    avg_recall = sum(recall_scores)/len(recall_scores)
    avg_f1 = sum(f1_score for f1_score in f1_scores if f1_score is not None)/sum(1 for f1_score in f1_scores if f1_score is not None)

    print(f'Average Precision : {avg_prec:.5f}')
    print(f'Average Recall : {avg_recall:.5f}')
    print(f'Average F1 : {avg_f1:.5f}')

    return {'precision': avg_prec, 'recall': avg_recall, 'f1': avg_f1}



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./data")
parser.add_argument("--output", type=str, default="./cluster_output")
parser.add_argument("--is_retrieval", action='store_true')
parser.add_argument("--dataset_type", type=str, default="test")
parser.add_argument("--cls_model_name", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    train_input_path = os.path.join(args.input, 'data_train.json')
    dev_input_path = os.path.join(args.input, 'data_dev.json')
    test_input_path = os.path.join(args.input, 'data_test.json')

    data_train = read_data(train_input_path)
    data_dev = read_data(dev_input_path)
    data_test = read_data(test_input_path)
    
    DATASET_TYPE = args.dataset_type
    RETRIEVAL = args.is_retrieval

    is_retrieval = "retrieval" if RETRIEVAL else "global"
    output_path = os.path.join(args.output, is_retrieval)
    os.makedirs(output_path, exist_ok=True)

    if DATASET_TYPE == 'dev':
        data = data_dev
    elif DATASET_TYPE == 'test':
        data = data_test
    elif DATASET_TYPE == 'train':
        data = data_train
    else:
        raise('Please set the correct dataset type (dev, test, train)')

    global_tweets = get_global_tweets(data)

    clusters = None
    if RETRIEVAL:
        clusters = retrieval_cluster(global_tweets, args.cls_model_name)
    else:
        clusters = global_cluster(global_tweets, args.cls_model_name)
    
    # Save clusters info
    save_cluster_info(clusters, output_path, DATASET_TYPE)

    gold_timelines = get_gold_timelines(data)
    clust_res = cluster_evaluate(gold_timelines, clusters, output_path, DATASET_TYPE)
    with open(os.path.join(output_path, 'cluster_score.json'), 'w') as f:
        json.dump(fp=f, obj=clust_res)


    
