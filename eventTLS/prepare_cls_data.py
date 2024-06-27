import os
import re
import json
import argparse


BINARY_CLASSIFICATION_PROMPT = """You are given a list of tweets, in chronological order, about some event in a timeline below:
 
{timeline_tweets}

Consider the following new tweet:

{new_tweet}

If this new tweet follows the given list of tweets in the same timeline (i.e., the new tweet is about the same event), then reply with “Yes”.
 
If the new tweet is not relevant to the event in the given timeline, then reply with “No, it is not relevant”.
 
If the new tweet is repetitive or redundant (i.e., it repeats information present in previous tweets in the given timeline), then reply with “No, it is repetitive”.
 
If a new tweet is not informative (i.e., it is generic or expresses some opinion but does not add new information to the event in the given timeline), then reply with “No, it is not informative”."""


def get_cls_data(process_tweets):
    yes_response = "Yes."
    no_rel_response = "No, it is not relevant."
    no_rep_response = "No, it is repetitive."
    no_info_response = "No, it is not informative."

    all_timeline_tweets_list, all_pred_tweet_list, all_labels, all_outputs = [], [], [], []

    for timeline in process_tweets:
        times = timeline['times']
        seed = timeline['seed']
        timeline_tweets = timeline['tweets']
        rej_reasons = timeline['majority_reasons']
        timeline_tweets.pop(0)
        seed_time = times.pop(0).strip('+00:00')
        timeline_labels = timeline['labels']
        context = []
        label_ids = []
        processed_timeline = []
        timeline_tweets_list, pred_tweet_list = [], []
        output = []

        assert len(timeline_tweets) == len(times)
        assert len(times) == len(rej_reasons)

        for time, tweet, label, rej in zip(times, timeline_tweets, timeline_labels, rej_reasons):
            time = time.strip('+00:00')
            clean_tweet = cleaning_tweet(time + ': ' + tweet)
            clean_seed = cleaning_tweet(seed_time + ': ' + seed)
            processed_timeline.append((clean_seed, " ".join(context +  [clean_tweet])))
            timeline_tweets_list.append([clean_seed] + context)
            pred_tweet_list.append(clean_tweet)

            label_ids.append(label2id_map[label])
            if label == '1':
                context.append(clean_tweet)
                output.append(yes_response)
            else:
                if 'informative' == rej or 'informative' in rej:
                    output.append(no_info_response)
                elif 'relevant' == rej or 'relevant' in rej:
                    output.append(no_rel_response)
                elif 'repetitive' == rej or 'repetitive' in rej:
                    output.append(no_rep_response)
                else:
                    output.append('No.')

        assert len(processed_timeline) == len(timeline_tweets) == len(label_ids) == len(output)
        all_timeline_tweets_list.extend(timeline_tweets_list)
        all_pred_tweet_list.extend(pred_tweet_list)
        all_labels.extend(label_ids)
        all_outputs.extend(output)

    assert len(all_labels) == len(all_pred_tweet_list) == len(all_timeline_tweets_list) == len(all_outputs)
    generated_cls_data = []
    for index in range(len(all_labels)):
        timeline_tweets, new_tweet, label = all_timeline_tweets_list[index], all_pred_tweet_list[index], all_labels[index]
        instruction = "You are given a list of tweets, in chronological order, about some event in a timeline below:"
        prompt_str = BINARY_CLASSIFICATION_PROMPT.format(
            timeline_tweets='\n'.join(timeline_tweets),
            new_tweet=new_tweet,
        )
        input_str = '\n'.join(prompt_str.split('\n')[1:]).strip()
        output_str = all_outputs[index]
        sample = {
            'instruction': instruction,
            'input': input_str,
            'output': output_str
        }
        generated_cls_data.append(sample)
    return generated_cls_data

label2id_map = {'0': 0, '1': 1}

def read_data(path):
    with open(path, mode='r') as f:
        data = json.loads(f.read())
    return data

def save_data(path, data):
    with open(path, 'w') as f:
        json.dump(obj=data, fp=f)

def cleaning_tweet(tweet):
    """Removes URLs from the tweet"""
    c_tweet = re.sub(r'http\S+', '', tweet).strip()
    c_tweet = re.sub('\s+', ' ', c_tweet).strip()
    return c_tweet


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./data")
parser.add_argument("--output", type=str, default="./data/cls")

if __name__ == "__main__":
    args = parser.parse_args()
    train_input_path = os.path.join(args.input, 'data_train.json')
    dev_input_path = os.path.join(args.input, 'data_dev.json')
    test_input_path = os.path.join(args.input, 'data_test.json')

    twt_train = read_data(train_input_path)
    twt_dev = read_data(dev_input_path)
    twt_test =  read_data(test_input_path)

    generated_cls_train = get_cls_data(twt_train)
    generated_cls_dev = get_cls_data(twt_dev)
    generated_cls_test = get_cls_data(twt_test)

    train_output_path = os.path.join(args.output, 'binary_train.json')
    dev_output_path = os.path.join(args.output, 'binary_dev.json')
    test_output_path = os.path.join(args.output, 'binary_test.json')

    os.makedirs(args.output, exist_ok=True)

    save_data(train_output_path, generated_cls_train)
    save_data(dev_output_path, generated_cls_dev)
    save_data(test_output_path, generated_cls_test)

    print(len(generated_cls_train), len(generated_cls_dev), len(generated_cls_test))

