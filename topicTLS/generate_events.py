import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta

import numpy as np
import openai

from tqdm import tqdm
from transformers import AutoTokenizer

from keywords_mapping import TARGET_KEYWORDS
from data import Dataset

from prompt_template import (ONESHOT_PROMPT_CRISIS, ONESHOT_PROMPT_CRISIS_TMP,
                             ONESHOT_PROMPT_ENTITIES, ONESHOT_PROMPT_ENTITIES_TMP,
                             ONESHOT_PROMPT_T17, ONESHOT_PROMPT_T17_TMP)



def completion_with_llm(prompt_str, temperature=0., max_len=512, stop_tokens=['4. '], model="meta-llama/Llama-2-13b"):
    completion = openai.Completion.create(model=model,
                                      prompt=prompt_str,
                                      max_tokens=max_len,
                                      temperature=temperature,
                                      stop=stop_tokens,
                                    )
    return completion

def get_target_corpus(corpus, keyword):
    target_corpus = [item for item in corpus if item['keyword'] == keyword]
    return target_corpus


parser = argparse.ArgumentParser()
parser.add_argument("--ds_path", type=str, default="./datasets/")
parser.add_argument("--dataset", type=str, default="entities")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b")
parser.add_argument("--extraction_path", type=str, default="./event_outputs")
parser.add_argument("--port", type=int, default=-1)
parser.add_argument("--temp", type=float, default=0.)
parser.add_argument("--context_length", type=int, default=1500)
parser.add_argument("--corpus_path", type=str, default="./corpus")
parser.add_argument("--keyword", type=str, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Corpus is all the articles of the dataset
    # Remove duplicate articles by title for entities.
    # Crisis & T17 datasets contains many articles without title. 
    corpus_path = os.path.join(args.corpus_path, args.dataset, 'articles.jsonl')
    corpus = []
    key_set = set()
    with open(corpus_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            key = item['title'] + item['date'] + item['keyword']
            if key in key_set and args.dataset == 'entities':
                continue
            corpus.append(item)
            key_set.add(key)
    print(f"Total corpus length: {len(corpus)}")

    template, tmp_template = None, None

    if args.dataset == 'entities':
        template = ONESHOT_PROMPT_ENTITIES
        tmp_template = ONESHOT_PROMPT_ENTITIES_TMP
    elif args.dataset == 'crisis':
        template = ONESHOT_PROMPT_CRISIS
        tmp_template = ONESHOT_PROMPT_CRISIS_TMP
    elif args.dataset == 't17':
        template = ONESHOT_PROMPT_T17
        tmp_template = ONESHOT_PROMPT_T17_TMP
    else:
        raise

    save_dir = None
    save_dir = os.path.join(args.extraction_path, args.dataset, args.model.split('/')[-1])
    openai.api_base = "http://0.0.0.0:8000/v1"  # use the IP or hostname of your instance
    openai.api_key = "none"  # vLLM server is not authenticated

    os.makedirs(save_dir, exist_ok=True)

    if args.port != -1:
        openai.api_base = f"http://0.0.0.0:{args.port}/v1"

    dataset_path = os.path.join(args.ds_path, args.dataset)
    dataset = Dataset(dataset_path)
    collections = dataset.collections
    print(f"{len(collections)} topics")

    for keyword, index in TARGET_KEYWORDS[args.dataset]:
        if args.keyword is not None:
            if keyword != args.keyword:
                continue

        kw_corpus = get_target_corpus(corpus, keyword)
        print(keyword, len(kw_corpus))
        save_path = os.path.join(save_dir, f"{keyword.replace(' ','_')}_events.jsonl")

        kw = ' '.join(collections[index].keywords)

        if args.dataset == 'entities':
            kw = keyword

        for item in tqdm(kw_corpus, desc=kw):
            title = item['title']
            date = item['date']
            content = item['content']
            tokens = tokenizer.encode(content)
            if len(tokens) > args.context_length:
                content = tokenizer.decode(tokens[:args.context_length], skip_special_tokens=True)
            prompt_str = None

            if keyword == 'mj':
                kw = 'Michael Jackson'
            if args.dataset == 'entities' and keyword == 'Bill Clinton':
                prompt_str = tmp_template.format(keyword=kw, content=content)
            elif args.dataset == 'crisis' and keyword == 'syria':
                prompt_str = tmp_template.format(keyword=kw, content=content)
            elif args.dataset == 't17' and keyword == 'iraq':
                prompt_str = tmp_template.format(keyword=kw, content=content)
            else:
                prompt_str = template.format(keyword=kw, content=content)
                
            completion = completion_with_llm(prompt_str=prompt_str, temperature=args.temp, max_len=256, stop_tokens=['#############', '\n\n', '\n4.', '4. '], model=model)
            output = completion['choices'][0]['text'].strip()

            if random.random() < 0.1:
                print(output)
            json_obj = {
                    'keyword': keyword,
                    'title': title,
                    'date': date,
                    'llm': output,
                    'index': item['index'],
            }
            with open(save_path, "a") as f:
                f.write(json.dumps(json_obj) + '\n')