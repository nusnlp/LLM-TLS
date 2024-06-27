import json
import re

separator = '\n####\n'

def get_the_best_two_workers(relevance_scores):
    workers_agreement = {'W1 W2': 0,
                         'W2 W3': 0,
                         'W1 W3': 0}

    # getting the best two workers
    for i, score in enumerate(relevance_scores):
        workers_votes = eval(score)
        w1_vs_w2 = 1 if workers_votes[0] == workers_votes[1] else 0
        w2_vs_w3 = 1 if workers_votes[1] == workers_votes[2] else 0
        w1_vs_w3 = 1 if workers_votes[0] == workers_votes[2] else 0

        workers_agreement['W1 W2'] += w1_vs_w2
        workers_agreement['W2 W3'] += w2_vs_w3
        workers_agreement['W1 W3'] += w1_vs_w3

    top_2_worker = max(workers_agreement, key=workers_agreement.get)
    best_w1, best_w2 = top_2_worker.split(' ')

    return best_w1, best_w2


def clean_text(text):
    """Removes URLs from the tweet"""
    c_text = re.sub(r'http\S+', '', text).strip()
    c_text = re.sub('\s+', ' ', c_text).strip()
    return c_text



def process_timelines_for_summarization(path, data, mode='oracle'):

    processed_timelines = []
    for timeline in data:
        batch_id = timeline['batch_id']
        timeline_id = timeline['timeline_id']
        seed = timeline['seed']
        tweets = timeline['tweets']
        relevance_scores = timeline['relevance_scores']
        timestamps = timeline['times']

        seed = clean_text(seed)
        tweets = [clean_text(tweet) for tweet in tweets]

        if mode == 'oracle':
            labels = timeline['labels']
        else:
            labels = timeline['predictions']

        assert len(tweets) == (len(timestamps))
        assert len(tweets) == (len(labels)+1)
        
        
        # get the total number of summaries
        summaries_num = [k.split()[1] for k in timeline.keys() if 'Summary' in k]

        if len(summaries_num) == 3:
            best_w1, best_w2 = get_the_best_two_workers(relevance_scores)

            # we want to train and eval models on the summaries
            # of the best two workers
            best_w1_summary = timeline[f'Summary {best_w1[1]}']
            best_w2_summary = timeline[f'Summary {best_w2[1]}']

            # summary_1 = timeline['Summary 1']
            # summary_2 = timeline['Summary 2']
            # summary_3 = timeline['Summary 3']

            relevant_tweets = [seed]
            relevant_tweets_time = [timestamps[0]]

            all_tweets = [seed]
            all_tweets_time = [timestamps[0]]


            for tweet, label, time in zip(tweets[1:], labels, timestamps[1:]):
                if int(label) == 1:
                    relevant_tweets.append(tweet)
                    relevant_tweets_time.append(time)
            
            for tweet, time in zip(tweets[1:], timestamps[1:]):
                all_tweets.append(tweet)
                all_tweets_time.append(time)
            


            processed_timeline = separator.join(relevant_tweets)

            for summary in [best_w1_summary, best_w2_summary]:
                processed_timelines.append({'batch_id': batch_id,
                                            'timeline_id': timeline_id,
                                            'timeline': processed_timeline,
                                            'tweets': relevant_tweets,
                                            'time': relevant_tweets_time,
                                            'all_tweets': all_tweets,
                                            'all_time': all_tweets_time,
                                            'summary': summary})

            # assert len(processed_timelines) == len(data) * 2

        else:
            summaries = [timeline[f'Summary {i}'] for i in summaries_num]

            relevant_tweets = [seed]
            relevant_tweets_time = [timestamps[0]]
            
            all_tweets = [seed]
            all_tweets_time = [timestamps[0]]


            for tweet, label, time in zip(tweets[1:], labels, timestamps[1:]):
                if int(label) == 1:
                    relevant_tweets.append(tweet)
                    relevant_tweets_time.append(time)
            
            for tweet, time in zip(tweets[1:], timestamps[1:]):
                all_tweets.append(tweet)
                all_tweets_time.append(time)
            
            

            processed_timeline = separator.join(relevant_tweets)

            for summary in summaries:
                processed_timelines.append({'batch_id': batch_id,
                                            'timeline_id': timeline_id,
                                            'timeline': processed_timeline,
                                            'tweets': relevant_tweets,
                                            'time': relevant_tweets_time,
                                            'all_tweets': all_tweets,
                                            'all_time': all_tweets_time,
                                            'label': labels,
                                            'summary': summary})

            # assert len(processed_timelines) == len(data) * 2


    with open(path, mode='w') as f:
        for timeline in processed_timelines:
            json.dump(timeline, f, ensure_ascii=False)
            f.write('\n')


def read_data(path):
    with open(path, mode='r') as f:
        data = json.load(fp=f)
    return data

if __name__ == '__main__':
    train_split_oracle = read_data('./data/data_train.json')
    dev_split_oracle = read_data('./data/data_dev.json')
    test_split_oracle = read_data('./data/data_test.json')

    process_timelines_for_summarization('./data/train.sum.json', train_split_oracle)
    process_timelines_for_summarization('./data/dev.sum.json', dev_split_oracle)
    process_timelines_for_summarization('./data/test.sum.json', test_split_oracle)


