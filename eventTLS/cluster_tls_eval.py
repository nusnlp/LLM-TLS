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

metric = load_metric("rouge")


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
    # decoded_preds = tokenizer.decode_batch(preds)
    # decoded_labels = tokenizer.decode_batch(labels)
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

    # eval_preds = tokenizer.encode_batch(new_preds), tokenizer.encode_batch(new_gold)
    eval_preds = tokenizer(new_preds)['input_ids'], tokenizer(new_gold)['input_ids']
    result = compute_metrics(eval_preds, tokenizer)

    return result

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

def find_temporal_center(timestamps):
    if isinstance(timestamps[0], datetime):
        parsed_timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    else:
        parsed_timestamps = [parse_timestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
    timestamps_in_seconds = [datetime.strptime(parsed_ts, "%Y-%m-%d %H:%M:%S").timestamp() for parsed_ts in parsed_timestamps]
    temporal_center = datetime.fromtimestamp(np.mean(timestamps_in_seconds))
    return temporal_center

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

def load_sum_model(sum_model_name, checkpoint_path, device='auto'):
    model = AutoModelForCausalLM.from_pretrained(sum_model_name , load_in_8bit=True , device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
    model = PeftModel.from_pretrained(model , checkpoint_path, device_map=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer



def sum_evaluate(gold_timelines, clusters, model, tokenizer, sum_data, output_path, DATASET_TYPE):
    gold2pred_mapping = get_gold_pred_mapping(gold_timelines, clusters)
    matched_pred_clusters = list(gold2pred_mapping.values())
    instruction = "You are given a starting event which is defined under ##Seed. You may be given incoming information related to the starting event under ##Timeline. Write a summary combining the starting event and the incoming information. If there is no incoming information given, summarize the starting event."

    samples = []
    for matched_pred_cluster_id in matched_pred_clusters:
        if matched_pred_cluster_id == -1: # do not generate summaries for unmatched gold clusters
            input_str = ""
        else:
            tweets = [twt['text'] for twt in clusters[matched_pred_cluster_id]]
            input_str = instruction + '\n\n'
            input_str += f"##Seed\n{tweets[0]}\n"
            if len(tweets) > 1:
                input_str += '\n##Timeline\n'
                cnt = 0
                for twt in tweets[1:]:
                    cnt += 1
                    input_str += f"{str(cnt)}. {twt}\n"
            
            input_str += '\n\n##Summary:\n'
        samples.append({
            'input': input_str,
        })
    with open("./tmp.json", 'w') as f:
        json.dump(fp=f, obj=samples)
    infer_samples = load_dataset('json' , data_files="./tmp.json")

    def gen_data():
        for i in infer_samples["train"]["input"]:
            yield { "train" : i }

    def tokenzied_examples(examples):
        return tokenizer(examples['train'] , truncation=True)

    raw_dataset = Dataset.from_generator(gen_data)
    tokenized_dataset = raw_dataset.map(tokenzied_examples , batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['train'])
    tokenized_dataset = tokenized_dataset.with_format("torch")

    batch_size=8

    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(tokenized_dataset , batch_size=batch_size , collate_fn=data_collator)

    inference_write_to = os.path.join(output_path, f'summary_{DATASET_TYPE}.txt')
    if os.path.exists(inference_write_to):
        os.remove(inference_write_to)
        
    with torch.no_grad():
        for step, batch in enumerate(tqdm(train_dataloader)):
            model_output = model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                early_stopping=True,
                do_sample=False,
                max_new_tokens=200,
            )
            decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)
            with open(inference_write_to , "a") as f:
                for i, x in enumerate(decoded_output):
                    raw_answer = x.strip()
                    if "##Summary:" not in raw_answer:
                        final_answer = ""
                    else:
                        final_answer = raw_answer.split("##Summary:\n")[1].split("\n")[0].strip()
                    final_answer = final_answer.encode('ascii', 'ignore').decode('ascii')
                    f.write(final_answer + "\n")
    timelineId2summary = {}

    for timeline in sum_data:
        timeline_id = timeline['batch_id'] + '_' + timeline['timeline_id']
        timeline_summary = timeline['summary']
        timelineId2summary.setdefault(timeline_id, []).append(timeline_summary)
    
    aligned_references = []

    gold_timeline_ids = list(gold2pred_mapping.keys())
    with open(inference_write_to, 'r+', encoding="utf-8") as f:
        predicted_summaries = f.readlines()
        for gold_timeline_id, predicted_summary in zip(gold_timeline_ids, predicted_summaries):
            summary = timelineId2summary[gold_timeline_id]
            aligned_references.append({
                'references' : [summary[0], summary[1]],
                'output' : predicted_summary,
            })
    
    aligned_inference_write_to = os.path.join(output_path, f'summary_{DATASET_TYPE}.jsonl')
    with open(aligned_inference_write_to, 'w') as f:
        for item in aligned_references:
            f.write(json.dumps({
                'output': item['output'],
                'references': item['references']
            }) + '\n')
    sum_res = rouge_eval(aligned_inference_write_to, tokenizer)
    return sum_res



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./data")
parser.add_argument("--output", type=str, default="./output")
parser.add_argument("--is_retrieval", action='store_true')
parser.add_argument("--dataset_type", type=str, default="test")
parser.add_argument("--sum_model_base", type=str)
parser.add_argument("--sum_model_lora_checkpoint", type=str)

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


    sum_data_dev = read_data(os.path.join(args.input, 'dev.sum.json'))
    sum_data_test = read_data(os.path.join(args.input, 'test.sum.json'))

    if DATASET_TYPE == 'dev':
        data = data_dev
    elif DATASET_TYPE == 'test':
        data = data_test
    elif DATASET_TYPE == 'train':
        data = data_train
    else:
        raise('Please set the correct dataset type (dev, test, train)')

    if DATASET_TYPE == 'dev':
        sum_data = sum_data_dev
    elif DATASET_TYPE == 'test':
        sum_data = sum_data_test

    clusters = []
    # retrieve the clusters from the saved json file

    file_name = os.path.join(output_path, f"cluster_{DATASET_TYPE}.json")

    with open(file_name, 'r') as f:
        clusters = json.load(f)

    timelineId2summary = {}
    for timeline in sum_data:
        timeline_id = timeline['batch_id'] + '_' + timeline['timeline_id']
        timeline_summary = timeline['summary']
        timelineId2summary.setdefault(timeline_id, []).append(timeline_summary)

    gold_timelines = get_gold_timelines(data)

    model, tokenizer = load_sum_model(args.sum_model_base, args.sum_model_lora_checkpoint, device='cuda')
    sum_res = sum_evaluate(gold_timelines, clusters, model, tokenizer, sum_data, output_path, DATASET_TYPE)
    with open(os.path.join(output_path, 'sum_scores.json'), 'w') as f:
        json.dump(fp=f, obj=sum_res)
