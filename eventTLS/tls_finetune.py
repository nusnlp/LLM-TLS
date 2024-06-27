from transformers import Trainer , TrainingArguments
from peft import LoraConfig , get_peft_model , prepare_model_for_int8_training , set_peft_model_state_dict , PeftModel
import torch
from torch.utils.data import Dataset
from typing import Dict , Optional , Sequence
from transformers import AutoModelForCausalLM , AutoTokenizer
import transformers
from datasets import load_dataset
from accelerate import Accelerator
import os 
import argparse
from transformers import TrainerCallback , TrainerState , TrainerControl , set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import re
import json
import copy
from dataclasses import dataclass, field

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
PREFIX_CHECKPOINT_DIR = "checkpoint"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name" , type=str , default="meta-llama/Llama-2-13b")
    parser.add_argument("--data" , type=str , default="./data/ft/tls_train.json")
    parser.add_argument("--output_dir" , type=str , required=True)
    parser.add_argument("--weight_decay" , default=0.05)
    parser.add_argument("--per_device_train_batch_size" , type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--num_epochs" , default=2)
    parser.add_argument("--warmup_steps" , default=5)
    parser.add_argument("--save_steps" , default=50)
    parser.add_argument("--save_limit" , default=2)
    
    #lora 
    parser.add_argument("--lora_r" , default=4)
    parser.add_argument("--lora_alpha" , default=16)
    parser.add_argument("--lora_dropout" , default=0.05)
    return parser.parse_args()

class SavePeftModelCallback(TrainerCallback):
    def on_save(self , args: TrainingArguments , state: TrainerState , control: TrainerControl , **kwargs):

        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


def smart_tokenizer_embedding_resize(special_tokens_dict: dict , tokenizer  , model):

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0 , keepdim=True)

        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0 , keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def format_output(s):
    return clean_text(s)

def format_input(data_point):
    seed_tweet = clean_text(data_point["seed"])
    tweets = data_point["tweets"]
    tweets.pop(0)
    labels = data_point["labels"]
    timelines = {}

    for i , j in enumerate(tweets):
        if labels[i] == '1':
            timelines[i] = clean_text(j)

    time_st = ""
    if timelines !={}:
        s = 1
        for i , j in timelines.items():
            time_st = time_st + str(s) + ". " + j + "\n"
            s = s + 1
    
    if time_st !="":
        st = f"""You are given a starting event which is defined under ##Seed. You may be given incoming information related to the starting event under ##Timeline.

Write a summary combining the starting event and the set of descriptions. If there is no information given , summarize the starting event.

##Seed:
{seed_tweet}

##Timeline:
{time_st}

##Summary:
"""
        
    else:
        st = f"""You are given a starting event which is defined under ##Seed. You may be given incoming information related to the starting event under ##Timeline.
        
Write a summary combining the starting event and the set of descriptions. If there is no information given , summarize the starting event.

##Seed:
{seed_tweet}

##Summary:
"""
        
    return st

def print_trainable_parameters(model):
    trainable_param = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()

    print(
        f"Trainable Parameters: {trainable_param} \nAll Parameters: {all_param}")

def clean_text(text):
    """Removes URLs from the tweet"""
    c_text = re.sub(r'http\S+', '', text).strip()
    c_text = re.sub('\s+', ' ', c_text).strip()
    return c_text

def read_data(path):
    with open(path, mode='r') as f:
        try:
            data = [json.loads(x) for x in f.readlines()]
        except:
            data = json.load(fp=f)
    return data


def _tokenize_fn(strings: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest",
                  max_length=2048, truncation=True)
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(input_ids=input_ids,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens
                )

def preprocess(sources: Sequence[str] , targets: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer)->Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    print("examples" , examples[0])
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    
    inputs_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(inputs_ids)
    lis = []
    final_labels = []
    for i , label, source_len in zip(inputs_ids , labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        lis.append(i)
        final_labels.append(label)

    return dict(input_ids = lis  , labels = final_labels)

class SuperVisedDataset(Dataset):
    def __init__(self , tokenizer , main_dataset):
        self.dataset = main_dataset

        targets = []
        sources = []
        s = 0

        for i in self.dataset:
            # inst = i["input"].split("\n\n##Summary:\n")[0]
            # l = i["input"].split("\n\n##Summary:\n")[1]
            # instruction = inst + "\n\n##Summary:\n"
            # output = l + tokenizer.eos_token

            instruction = i['input']
            output = i['output'] + tokenizer.eos_token


            sources.append(instruction)
            targets.append(output)

        data_dict = preprocess(sources , targets , tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self , instances: Sequence[Dict]) -> Dict[str , torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )

def make_supervised_data(tokenizer , args):
    data_path = args.data

    dataset_tls = load_dataset("json" , data_files=data_path)

    train_dataset = SuperVisedDataset(tokenizer = tokenizer , main_dataset=dataset_tls["train"])

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

    


def train(args):
    print(f"*****Running on model: {args.model_name}*****")

    print(f"per_device_train_batch_size is {args.per_device_train_batch_size}, accumulation steps is {args.accumulate}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.accumulate,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        logging_steps=1,
        ddp_find_unused_parameters=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_limit,
        load_best_model_at_end=False
    )
    
    print("EOS Token" , tokenizer.eos_token)
    print("BOS Token" , tokenizer.bos_token)
    print("unk Token" , tokenizer.unk_token)
    
    # special_tokens_dict = {}
    # special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN    
    model = AutoModelForCausalLM.from_pretrained(args.model_name , load_in_8bit=True, use_cache=True,
    device_map={ "": Accelerator().process_index})
    model.config.pad_token_id = model.config.eos_token_id

    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model , lora_config)

    print_trainable_parameters(model)

    # smart_tokenizer_embedding_resize(special_tokens_dict=special_tokens_dict , tokenizer=tokenizer , model=model)

    data_module = make_supervised_data(tokenizer = tokenizer , args = args)

    print("train dataset value" , data_module["train_dataset"][0])

    trainer = Trainer(model=model, tokenizer=tokenizer, callbacks=[SavePeftModelCallback],
                     args=training_args, **data_module)
    trainer.train()
     #rainer.save_state()
    model.save_pretrained(os.path.join(args.output_dir, 'final_checkpoint'))

if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    os.makedirs(args.output_dir , exist_ok=True)
    train(args)