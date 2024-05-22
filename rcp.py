#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

import tqdm

from reading_comprehension_dataset import ReadingComprehensionDataset
from trainable_module import TrainableModule

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--uepochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(TrainableModule):
    def __init__(self, args, robeczech):
        super().__init__()

        self.backbone = robeczech
        self.simple = torch.nn.Linear(robeczech.config.hidden_size, 2)

        # freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        
        x = self.backbone(input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1)).last_hidden_state
        return self.simple(x)


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

    def create_dataset(dataset, is_test=False):
        data = []
        i = 0
        for elem in dataset:
            for x in elem['qas']:

                context, question = elem['context'], x['question']

                tokens = tokenizer(
                        context, question,
                        return_tensors='pt',
                        padding='max_length',
                        max_length=512,
                        truncation='only_first'
                    )

                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']

                if is_test:
                    data.append((
                        {'tokens': tokens, 'context': context},
                        {'start': 0, 'end': 0}
                    ))
                    continue

                start = x['answers'][0]['start']
                end = start + len(x['answers'][0]['text']) - 1

                token_start = tokens.char_to_token(start)
                token_end = tokens.char_to_token(end)

                if token_start is None or token_end is None:
                    #i += 1
                    continue

                data.append((
                    {'tokens': tokens, 'context': context},
                    {'start': token_start, 'end': token_end}
                    #targets
                ))

        #print(f"skipped {i} examples")
        #exit()
        return data


    def prepare_batch(data):
        inputs, targets = zip(*data)
        inputs = [item['tokens'] for item in inputs]

        targets = torch.tensor([
            [x['start'], x['end']] for x in targets
        ])

        inputs_ids = torch.stack([x['input_ids'] for x in inputs])
        attention_mask = torch.stack([x['attention_mask'] for x in inputs])

        return (inputs_ids, attention_mask), targets 


    # Load the data
    dataset = ReadingComprehensionDataset()

    train = create_dataset(dataset.train.paragraphs)
    dev = create_dataset(dataset.dev.paragraphs)
    test = create_dataset(dataset.test.paragraphs, is_test=True)

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, collate_fn=prepare_batch)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, collate_fn=prepare_batch)


    print("created dataloaders")

    # TODO: Create the model and train it
    model = Model(args, robeczech)

    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01),
        loss=torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        ),
        metrics={"accuracy": torchmetrics.Accuracy(
             "multiclass",
             num_classes=512,
             ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        )},
    )

    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": 5e-5},
    ]

    model.configure(
        optimizer=torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5, weight_decay=0.01),
        loss=torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        ),
        metrics={"accuracy": torchmetrics.Accuracy(
             "multiclass",
             num_classes=512,
             ignore_index=tokenizer.convert_tokens_to_ids('[PAD]')
        )},
    )

    train_dataloader = torch.utils.data.DataLoader(train, collate_fn=prepare_batch, shuffle=True, batch_size=8)
    dev_dataloader = torch.utils.data.DataLoader(dev, collate_fn=prepare_batch, shuffle=False, batch_size=8)
    model.fit(train_dataloader, dev=dev_dataloader, epochs=args.uepochs)

    predictions = model.predict(test_loader)
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        
        for i in range(len(test)):
            context = test[i][0]['context']
            tokens = test[i][0]['tokens']
            
            start_logits_batch, end_logits_batch = predictions[i][:, 0], predictions[i][:, 1]

            pred_start = np.argmax(start_logits_batch)
            pred_end = np.argmax(end_logits_batch)

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

