#!/usr/bin/env python3

# 21e65bb3-23db-11ec-986f-f39926f24a9c
# e4553c7b-e907-46c6-98e5-f08d1ce8f040

import argparse
import datetime
import os
import re

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
import torchmetrics
import transformers
from transformers import get_linear_schedule_with_warmup

from reading_comprehension_dataset import ReadingComprehensionDataset
from trainable_module import TrainableModule

from pprint import pprint as pp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=14, type=int, help="Number of epochs.")
parser.add_argument("--uepochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")


class Example:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers, tokenizer, test):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.tokenizer = tokenizer
        self.test = test
        self.token = None

    def preprocess(self):
        context = " ".join(self.context.split())
        question = " ".join(self.question.split())
        answer = " ".join(self.answer_text.split())

        end_char_idx = self.start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        is_char_in_ans = [0] * len(context)
        for idx in range(self.start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        tokenized_context = self.tokenizer(context, return_offsets_mapping=True)

        self.token = tokenized_context

        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context['offset_mapping']):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)
        
        if len(ans_token_idx) and not self.test == 0:
            self.skip = True
            return

        if len(ans_token_idx)== 0 and self.test:
            start_token_idx = -1
            end_token_idx = -1

        else:
            start_token_idx = ans_token_idx[0]
            end_token_idx = ans_token_idx[-1]

        tokenized_question = self.tokenizer(question)


        input_ids = tokenized_context['input_ids'] + tokenized_question['input_ids'][1:]
        token_type_ids = [0] * len(tokenized_context['input_ids']) + [1] * len(tokenized_question['input_ids'][1:])
        attention_mask = [1] * len(input_ids)

        # Truncate if necessary
        if len(input_ids) > 512:
            input_ids = input_ids[:512]
            attention_mask = attention_mask[:512]
            token_type_ids = token_type_ids[:512]

            if start_token_idx >= 512 or end_token_idx >= 512 and self.test:
                start_token_idx = -1
                end_token_idx = -1
            else:
                self.skip = True
                return

        # Pad if necessary
        padding_length = 512 - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.token = tokenized_context



def create_squad_examples(raw_data, tokenizer, test=False):
    squad_examples = []
    #print(raw_data)
    for item in raw_data:
        context = item["context"]
        for qa in item["qas"]:
            question = qa["question"]
            
            
            if test:
                start_char_idx = 0
                answer_text = ""
                all_answers = [""]
            else:
                start_char_idx = qa["answers"][0]["start"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]] 
            
            squad_eg = Example(question, context, start_char_idx, answer_text, all_answers, tokenizer, test)
            squad_eg.preprocess()
            squad_examples.append(squad_eg)
    
    return squad_examples

def create_inputs_targets(examples):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in examples:
        if not item.skip:
            dataset_dict["input_ids"].append(item.input_ids)
            dataset_dict["attention_mask"].append(item.attention_mask)
            dataset_dict["start_token_idx"].append(item.start_token_idx)
            dataset_dict["end_token_idx"].append(item.end_token_idx)

    for key in dataset_dict:
        #print(torch.tensor(dataset_dict[key]).shape)
        dataset_dict[key] = torch.tensor(dataset_dict[key])
    
    inputs = {
        "input_ids": dataset_dict["input_ids"],
        "attention_mask": dataset_dict["attention_mask"]
        #"tokenizer": dataset_dict["tokenizer"]
    }
    targets = torch.tensor(list(zip(dataset_dict["start_token_idx"], dataset_dict["end_token_idx"])))

    return inputs, targets

class Model(TrainableModule):
    def __init__(self, args, robeczech):
        super().__init__()

        self.backbone = robeczech
        self.output_start = torch.nn.Linear(robeczech.config.hidden_size, 1)
        self.output_end = torch.nn.Linear(robeczech.config.hidden_size,1)

        # freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:

        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)

        x = outputs.last_hidden_state
        x_start = self.output_start(x).squeeze(-1)
        x_end = self.output_end(x).squeeze(-1)

        return x_start, x_end
    

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the pre-trained RobeCzech model
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    special_tokens_dict = {"additional_special_tokens": [".", ","]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    robeczech.resize_token_embeddings(len(tokenizer))

    dataset = ReadingComprehensionDataset()
    

    print("creating train examples")
    train_squad_examples = create_squad_examples(dataset.train.paragraphs, tokenizer)
    print("finished train examples")
    #x_train, y_train = create_inputs_targets(train_squad_examples)

    print("creating dev examples")
    dev_squad_examples = create_squad_examples(dataset.dev.paragraphs, tokenizer)
    print("finished dev examples")
    #x_dev, y_dev = create_inputs_targets(dev_squad_examples)

    print("creating test examples")
    test_squad_examples = create_squad_examples(dataset.test.paragraphs, tokenizer, test=True)
    print("finisehd test examples\n\n\n")
    #x_test, y_test = create_inputs_targets(test_squad_examples)

    train_dataloader = torch.utils.data.DataLoader(train_squad_examples, batch_size=args.batch_size, shuffle=True, collate_fn=create_inputs_targets)
    dev_dataloader = torch.utils.data.DataLoader(dev_squad_examples, batch_size=args.batch_size, shuffle=False, collate_fn=create_inputs_targets)
    test_dataloader = torch.utils.data.DataLoader(test_squad_examples, batch_size=1, shuffle=False, collate_fn=create_inputs_targets)

    model = Model(args, robeczech)
    
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": 5e-5},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'))

    model.configure(optimizer=optimizer, loss=loss_fn)

    print("configured model")

    # Training
    model.fit(train_dataloader, dev=dev_dataloader, epochs=args.epochs)

    for param in model.backbone.parameters():
        param.requires_grad = True

    # Reconfigure optimizer and scheduler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.uepochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model.configure(optimizer=optimizer, loss=loss_fn)

    train_dataloader = torch.utils.data.DataLoader(train_squad_examples, batch_size=4, shuffle=True, collate_fn=create_inputs_targets)
    dev_dataloader = torch.utils.data.DataLoader(dev_squad_examples, batch_size=4, shuffle=False, collate_fn=create_inputs_targets)

    # Continue training with unfrozen backbone
    model.fit(train_dataloader, dev=dev_dataloader, epochs=args.uepochs)

    # Evaluation
    print('Predicting...')
    predictions = model.predict(test_dataloader, as_numpy=True)

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        for i, (_, _) in enumerate(test_dataloader):
            #context = test[i][0]['context']
            context = train_squad_examples[i].context
            start_logits_batch, end_logits_batch = predictions[i]
            pred_start = np.argmax(start_logits_batch)
            pred_end = np.argmax(end_logits_batch)

            tokens = train_squad_examples[i].token
            
            if tokens is not None:
                start_char_info = tokens.token_to_chars(0, pred_start)
                end_char_info = tokens.token_to_chars(0, pred_end)
                if start_char_info is not None and end_char_info is not None:
                    start_char = start_char_info.start
                    end_char = end_char_info.end
                    answer_text = context[start_char:end_char]
                    print(answer_text)
                    print(answer_text, file=predictions_file)
                else:
                    print("", file=predictions_file)
            else:
                print("", file=predictions_file)
    total_predictions = sum(1 for line in open(os.path.join(args.logdir, "reading_comprehension.txt")))
    print(f"Total predictions made: {total_predictions}")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

