
from data import Dataset
from keywords_mapping import TARGET_KEYWORDS
import os
from tqdm import tqdm
import datetime
import json
import argparse

def get_dates_between(start_date, end_date):
    if end_date < start_date:
        raise ValueError("End date must be after start date.")
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.date())
        current_date += datetime.timedelta(days=1)
    return dates



parser = argparse.ArgumentParser()
parser.add_argument("--ds_path", type=str, default="./datasets/")
parser.add_argument("--dataset", type=str, default="entities")
parser.add_argument("--save_path", type=str, default="./corpus")

if __name__ == '__main__':
    args = parser.parse_args()
    ds_name = args.dataset

    dataset_path = os.path.join(args.ds_path, ds_name)
    dataset = Dataset(dataset_path)
    collections = dataset.collections
    save_path = args.save_path

    os.makedirs(os.path.join(save_path, ds_name), exist_ok=True)

    for keyword, index in TARGET_KEYWORDS[ds_name]:
        print(keyword)

        col = collections[index]
        assert col.name.replace('_', ' ') == keyword
        articles = [a for a in col.articles()]

        kw = ' '.join(col.keywords)

        kw_corpus_objects = []
        for idx, a in tqdm(enumerate(articles), desc=kw):
            title = a.title.strip() if a.title else ""
            
            sentences = a.sentences
            sentence_with_time = []
            for s in sentences:
                try:
                    if s.time is not None:
                        if s.time_level == 'd':
                            date = datetime.datetime.strptime(str(s.time)[:10], '%Y-%m-%d').date()
                            sentence_with_time.append(f"{str(date)}: {s.raw.strip()}")
                        elif s.time_level == 'm':
                            start, end = s.time[0], s.time[1]
                            dates_between = get_dates_between(start, end)
                            for d in dates_between:
                                sentence_with_time.append(f"{str(d)}: {s.raw.strip()}")
                except Exception as e:
                    print(e)

            pb_date = datetime.datetime.strftime(a.time, '%Y-%m-%d')
            content = f"Publish Date: {pb_date}\nContent:\n"
            if title:
                content = f"Title: {title}\nPublish Date: {pb_date}\nContent:\n"
            
            for s in sentences:
                s_raw = s.raw
                if s.time and s.time_level == 'd':
                    sent_time = datetime.datetime.strftime(s.time, '%Y-%m-%d')
                    content += f"{s_raw}({sent_time}) \n"
                else:
                    content += f"{s_raw} "
            
            article_obj = {
                'keyword': keyword,
                'index': idx,
                'kw': kw,
                'title': title,
                'date': pb_date,
                'content': content,
                'sentence_with_time': sentence_with_time,
            }
            kw_corpus_objects.append(article_obj)
        
        corpus_save_path = os.path.join(save_path, ds_name, 'articles.jsonl')
        
        with open(corpus_save_path, 'a') as f:
            for item in kw_corpus_objects:
                f.write(json.dumps(item) + '\n')




        