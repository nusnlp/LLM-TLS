# LLM-TLS
Repo for paper "From Moments to Milestones: Incremental Timeline Summarization Leveraging Large Language Models". 

Paper link: https://aclanthology.org/2024.acl-long.390.pdf

## Event-TLS

### Dataset
The tweet ID and summary data we used in ```./data``` is from [CrisisLTLSum](https://github.com/CrisisLTLSum/CrisisTimelines). Due to tweet privacy policy, we cannot release the content of the tweet. We recommend to download the tweet content through ```https://www.tweepy.org/``` using the tweet ID.

### Workflow
1. To prepare data for tweet timeline membership classification training, please refer to ```prepare_cls_data.py```

```
python prepare_cls_data.py --input "./data" --output "./data/cls"
```

2. To prepare data for timeline summarization training, please refer to ```preprocess_timelines.py``` and ```prepare_tls_data.py```

```
python preprocess_timelines.py
python prepare_tls_data.py --input "./data" --output "./data/ft"
```

In our implementation, we host the classification model through vLLM using the below command in a separate process:
```
python -m vllm.entrypoints.openai.api_server --model "CLS MODEL PATH" --port 8000
```

3. Perform incremental clustering process by ```generate_clusters.py```
```
python generate_clusters.py --input "./data" --output "./cluster_output" --dataset_type "test" --cls_model_name "[CLS MODEL PATH]" --is_retrieval
```


4. Perform timeline summarization and evaluation by ```cluster_tls_eval.py```
```
python cluster_tls_eval.py --input "./data" --output "./cluster_output" --dataset_type "test" --sum_model_base "SUMMARIZATION MODEL PATH" --sum_model_lora_checkpoint "LORA CHECKPOINT PATH" --is_retrieval
```

---

## Topic-TLS

### Download Dataset
To download datasets(T17, Crisis, Entities), please refer to [complementizer/news-tls ](https://github.com/complementizer/news-tls).

### Workflow
1. To preprocess dataset articles to certain format, please refer to ```preprocess_articles.py```
```
python preprocess_articles.py --ds_path "./datasets" --dataset "entities" --save_path "./corpus"
```

In our implementation, we host the model through vLLM using the below command in a separate process:
```
python -m vllm.entrypoints.openai.api_server --model "meta-llama/Llama-2-13b-hf" --port 8000
```

2. Perform LLM for event generation by ```generate_events.py```.
```
python generate_events.py --dataset entities --model "meta-llama/Llama-2-13b-hf" --extraction_path "./event_outputs"
```


3. Perform incremental clustering process by ```generate_clusters.py```
```
python generate_clusters.py \
    --model meta-llama/Llama-2-13b-hf \
    --output ./timelines_output \
    --input ./event_outputs/entities/Llama-2-13b-hf \
    --top_n 20 \
    --dataset entities \
    --incremental \
    --keyword all
```

4. Perform timeline summarization and evaluation by ```cluster_tls_eval.py```
```
python cluster_tls_eval.py \
    --timelines_path ./timelines_output/entities/Llama-2-13b-hf \
    --events_path ./event_outputs/entities/Llama-2-13b-hf \
    --output ./result/entities \
    --dataset entities \
    --text_rank
```

If you encounter any problem with the code running, please contact qishenghu@u.nus.edu .

---
# Citation

If you make use of our released source code, please cite the following paper:
 
> Qisheng Hu, Geonsik Moon, Hwee Tou Ng (2024). From Moments to Milestones: Incremental Timeline Summarization Leveraging Large Language Models. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024). (pp. 7232 – 7246). Bangkok, Thailand.


---

# License

The source code in this repository is licensed under the GNU General Public License Version 3. For commercial use of this code, separate commercial licensing is also available. Please contact:

* Qisheng Hu (qishenghu@u.nus.edu) 
* Hwee Tou Ng (nght@comp.nus.edu.sg)