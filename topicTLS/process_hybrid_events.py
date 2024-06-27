import argparse
import json
import os
import pickle
import random
import time
import warnings
from datetime import datetime, timedelta

from keywords_mapping import TARGET_KEYWORDS

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b")
parser.add_argument("--input", type=str, default="./event_outputs")
parser.add_argument("--corpus_path", type=str, default="./corpus")
parser.add_argument("--output", type=str, default="./events_hybrid")
parser.add_argument("--dataset", type=str, default='entities')
parser.add_argument("--use_sentences", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    output_path = os.path.join(args.output, args.dataset, args.model.split('/')[-1])
    input_path = os.path.join(args.input, args.dataset, args.model.split('/')[-1])

    os.makedirs(output_path, exist_ok=True)

    corpus_dict = {}
    if args.use_sentences:
        article_path = os.path.join(args.corpus_path, args.dataset, 'articles.jsonl')
        corpus = [json.loads(x) for x in open(article_path, 'r')]
        for item in corpus:
            keyword = item['keyword']
            index = item['index']
            if keyword not in corpus_dict:
                corpus_dict[keyword] = {}
            corpus_dict[keyword][index] = item
    
    for keyword, index in TARGET_KEYWORDS[args.dataset]:

        save_path = os.path.join(output_path, f"{keyword.replace(' ', '_')}_events.jsonl")

        read_path = os.path.join(input_path, f"{keyword.replace(' ', '_')}_events.jsonl")
        assert os.path.exists(read_path), f'{read_path} not exist'

        # Load generated events summaries
        with open(read_path, "r") as f:
            data = [json.loads(x) for x in f]
        data = [item for item in data if 'llm' in item or 'summary' in item]
        assert keyword == data[0]['keyword']

        keyword_corpus = corpus_dict[keyword]
        keyword_list = keyword_corpus[0]['kw'].split(' ') + [keyword]
        keyword_list = [i.strip().lower() for i in keyword_list ]
        events = []
        for item in data:
            events.append(item)

            e_index = item['index']
            article_obj = keyword_corpus[e_index]
            sentences_with_time = article_obj['sentence_with_time']
            for s in sentences_with_time:
                events.append({
                    'keyword': keyword,
                    'date': article_obj['date'],
                    'title': article_obj['title'],
                    'sentence': s.strip(),
                })
        with open(save_path, 'w') as f:
            for item in events:
                f.write(json.dumps(item) + '\n')