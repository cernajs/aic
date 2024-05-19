#!/usr/bin/env python3

# 21e65bb3-23db-11ec-986f-f39926f24a9c
# e4553c7b-e907-46c6-98e5-f08d1ce8f040

import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

from reading_comprehension_dataset import ReadingComprehensionDataset
from trainable_module import TrainableModule

from pprint import pprint as pp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs", default=4, type=int, help="Number of epochs.")
parser.add_argument("--uepochs", default=4, type=int, help="Number of unfreezed epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")


class Model(TrainableModule):
    def __init__(self, args, robeczech):
        super().__init__()

        self.backbone = robeczech
        self.hidden = torch.nn.Linear(robeczech.config.hidden_size, 1024)
        self.dropout = torch.nn.Dropout(0.4)
        self.output_start = torch.nn.Linear(1024, 1)
        self.output_end = torch.nn.Linear(1024,1)
        self.relu = torch.nn.ReLU()

        self.simple = torch.nn.Linear(robeczech.config.hidden_size, 2)

        # freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x) -> torch.Tensor:

        x = self.backbone(x['input_ids'], attention_mask=x['attention_mask']).last_hidden_state
        #x = self.relu(self.hidden(x))
        #x = self.dropout(x)
        #x = self.output(x)
        
        #x = self.simple(x)
            
        x = self.relu(self.hidden(x))
        x = self.dropout(x)
        x_start = self.output_start(x).squeeze(-1)
        x_end = self.output_end(x).squeeze(-1)

        return x_start, x_end

        #return x

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

    def create_dataset(dataset, is_test=False):
        """Create dataset from ReadingComprehensionDataset in tuple form."""
        data = []

        for elem in tqdm(dataset):
            for x in elem['qas']:
                context, question = elem['context'], x['question']

                tokens = tokenizer(
                    context, question,
                    return_tensors='pt',
                    padding=True,
                    max_length=512,
                    truncation='only_first'
                )

                if is_test:
                    data.append((
                        {'context' : context, 'question': question},
                        {'start': 0, 'end': 0}
                    ))
                    continue

                start = x['answers'][0]['start']
                end = start + len(x['answers'][0]['text'])

                if tokens.char_to_token(start) is None or tokens.char_to_token(end) is None:
                    continue

                data.append((
                    {'context' : context, 'question': question},
                    {'start': start, 'end': end}
                ))

        return data

    def prepare_batch(data):
        """Prepare batch for the model used in collate_fn."""
        inputs, targets = zip(*data)

        cx = [x['context'] for x in inputs]
        q = [x['question'] for x in inputs]

        tokens = tokenizer(
            cx, q,
            return_tensors='pt',
            padding=True,
            max_length=512,
            truncation='only_first',
        )

        targets = torch.tensor([
            (tokens.char_to_token(i, x['start']), tokens.char_to_token(i, x['end']))
            for i, x in enumerate(targets)
        ])

        #return tokens, (tokens, targets)
        return tokens, targets

    def create_dataloader(data, shuffle=False, batch_size=args.batch_size):
        """Create DataLoader from dataset."""

        return torch.utils.data.DataLoader(
            data,
            batch_size,
            shuffle=shuffle,
            collate_fn=prepare_batch,
            # num_workers=8, persistent_workers=True
        )

    # Create dataloaders
    print('Creating dataloaders...')
    train = create_dataset(dataset.train.paragraphs)
    dev = create_dataset(dataset.dev.paragraphs)
    test = create_dataset(dataset.test.paragraphs, is_test=True)

    train_dataloader = create_dataloader(train, shuffle=True)
    dev_dataloader = create_dataloader(dev, shuffle=False)
    test_dataloader = create_dataloader(test, shuffle=False, batch_size=1)
    """
    j = 0:
    for i in test:
        print(i)
        if j == 25:
            break
        j += 1
    """
    """ 
    test_idx = 0
    for i, (tokens, inputs) in enumerate(test_dataloader):
        batch_context = []
        for j in range(test_idx, test_idx + 3):
            batch_context.append(test[j][0]["context"])

        print(batch_context)
        test_idx += 3

        print(i)
        print(tokens)
        #print(inputs) #not-important
        #print(batch_context)
        
        print("\n\n")
        if i == 10:
            exit()
    """
    model = Model(args, robeczech)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
        loss=torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        ),
        # metrics={"accuracy": torchmetrics.Accuracy(
        #     "multiclass",
        #     num_classes=len(tokenizer.get_vocab()) - 1,
        #     ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        # )},
    )
    
    model.fit(train_dataloader, dev=dev_dataloader, epochs=args.epochs)

    for param in model.backbone.parameters():
        param.requires_grad = True

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
        loss=torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        ),
        # metrics={"accuracy": torchmetrics.Accuracy(
        #     "multiclass",
        #     num_classes=len(tokenizer.get_vocab()) - 1,
        #     ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        # )},
    )

    model.fit(train_dataloader, dev=dev_dataloader, epochs=args.uepochs)
    
    j = 0
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        predictions = model.predict(test_dataloader)

        for i, (tokens, inputs) in enumerate(test_dataloader):
            context = test[i][0]['context']
            """
            print(f"{j} {context}")
            
            print(f"tokens {tokens}, inputs {inputs}")
            j += 1
            if j == 20:
                exit()
            """
            #pred_start, pred_end = predictions[i][:, 0], predictions[i][:, 1]

            start_logits_batch, end_logits_batch = predictions[i]
            
            #print(f"start_logits_batch {start_logits_batch}")
            #print(f"end_logits_batch {end_logits_batch}")

            pred_start = np.argmax(start_logits_batch)
            pred_end = np.argmax(end_logits_batch)

            #pred_start = np.argmax(pred_start)
            #pred_end = np.argmax(pred_end)
        
            #0 bcs batch 1
            start_char_info = tokens.token_to_chars(0, pred_start)
            end_char_info = tokens.token_to_chars(0, pred_end)

            if start_char_info is not None and end_char_info is not None:
                start_char = start_char_info.start
                end_char = end_char_info.end
                answer_text = context[start_char:end_char]
                print(answer_text, file=predictions_file)
            else:
                print("", file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

