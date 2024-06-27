import os
import re
import json
import argparse
from datetime import datetime

def parse_timestamp(timestamp):
    if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2}', timestamp):
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S%z')
    elif re.match(r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}', timestamp):
        return datetime.strptime(timestamp, '%m/%d/%Y, %H:%M:%S')
    else:
        raise ValueError(f'Unknown timestamp format: {timestamp}')
    
def read_data(path):
    with open(path, mode='r') as f:
        data = [json.loads(x) for x in f]
    return data

def save_data(path, data):
    with open(path, 'w') as f:
        json.dump(obj=data, fp=f)

def cleaning_tweet(tweet):
    """Removes URLs from the tweet"""
    c_tweet = re.sub(r'http\S+', '', tweet).strip()
    c_tweet = re.sub('\s+', ' ', c_tweet).strip()
    return c_tweet

def get_ft_data(data):
    instruction = "You are given a starting event which is defined under ##Seed. You may be given incoming information related to the starting event under ##Timeline. Write a summary combining the starting event and the incoming information. If there is no incoming information given, summarize the starting event."

    ft_data = []
    for item in data:
        input_str = instruction + '\n\n'
        output_str = item['summary'].strip()
        tweets, times = item['tweets'], item['time']
        assert len(tweets) == len(times)
        seed_time = parse_timestamp(times[0]).strftime('%Y-%m-%d %H:%M:%S')
        input_str += f"##Seed\n{seed_time}: {tweets[0]}\n"
        if len(tweets) > 1:
            input_str += '\n##Timeline\n'
            cnt = 0
            for twt, ti in zip(tweets[1:], times[1:]):
                cnt += 1
                timestamp = parse_timestamp(ti).strftime('%Y-%m-%d %H:%M:%S')
                input_str += f"{str(cnt)}. {timestamp}: {twt}\n"
        input_str += '\n\n##Summary:\n'
        ft_data.append({
            'input': input_str,
            'output': output_str
        })
    return ft_data

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./data")
parser.add_argument("--output", type=str, default="./data/ft")

if __name__ == "__main__":
    args = parser.parse_args()
    train_input_path = os.path.join(args.input, 'train.sum.json')
    dev_input_path = os.path.join(args.input, 'dev.sum.json')
    test_input_path = os.path.join(args.input, 'test.sum.json')

    train = read_data(train_input_path)
    dev = read_data(dev_input_path)
    test =  read_data(test_input_path)

    generated_ft_train = get_ft_data(train)
    generated_ft_dev = get_ft_data(dev)
    generated_ft_test = get_ft_data(test)

    train_output_path = os.path.join(args.output, 'tls_train.json')
    dev_output_path = os.path.join(args.output, 'tls_dev.json')
    test_output_path = os.path.join(args.output, 'tls_test.json')

    os.makedirs(args.output, exist_ok=True)

    save_data(train_output_path, generated_ft_train)
    save_data(dev_output_path, generated_ft_dev)
    save_data(test_output_path, generated_ft_test)

    print(len(generated_ft_train), len(generated_ft_dev), len(generated_ft_test))

