#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchaudio.models.decoder
import torchmetrics

from common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class TrainableModule(torch.nn.Module):
    """A simple Keras-like module for training with raw PyTorch.

    The module provides fit/evaluate/predict methods, computes loss and metrics,
    and generates both TensorBoard and console logs. By default, it uses GPU
    if available, and CPU otherwise. Additionally, it offers a Keras-like
    initialization of the weights.

    The current implementation supports models with either single input or
    a tuple of inputs; however, only one output is currently supported.
    """
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    from time import time as _time
    from tqdm import tqdm as _tqdm

    def configure(self, *, optimizer=None, schedule=None, loss=None, metrics={}, logdir=None, device="auto"):
        """Configure the module process.

        - `optimizer` is the optimizer to use for training;
        - `schedule` is an optional learning rate scheduler used after every batch;
        - `loss` is the loss function to minimize;
        - `metrics` is a dictionary of additional metrics to compute;
        - `logdir` is an optional directory where TensorBoard logs should be written;
        - `device` is the device to use; when "auto", `cuda` is used when available, `cpu` otherwise.
        """
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss, self.loss_metric = loss, torchmetrics.MeanMetric()
        self.metrics = torchmetrics.MetricCollection(metrics)
        self.logdir, self._writers = logdir, {}
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.to(self.device)

    def load_weights(self, path, device="auto"):
        """Load the model weights from the given path."""
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save_weights(self, path):
        """Save the model weights to the given path."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        """Train the model on the given dataset.

        - `dataloader` is the training dataset, each element a pair of inputs and an output;
          the inputs can be either a single tensor or a tuple of tensors;
        - `dev` is an optional development dataset;
        - `epochs` is the number of epochs to train;
        - `callbacks` is a list of callbacks to call after each epoch with
          arguments `self`, `epoch`, and `logs`;
        - `verbose` controls the verbosity: 0 for silent, 1 for persistent progress bar,
          2 for a progress bar only when writing to a console.
        """
        for epoch in range(epochs):
            self.train()
            self.loss_metric.reset()
            self.metrics.reset()
            start = self._time()
            epoch_message = f"Epoch={epoch+1}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if verbose == 2 else not verbose)
            for xs, y in data_and_progress:
                xs, y = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(self.device)
                logs = self.train_step(xs, y)
                message = [epoch_message] + [f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()]
                data_and_progress.set_description(" ".join(message), refresh=False)
            if dev is not None:
                logs |= {"dev_" + k: v for k, v in self.evaluate(dev, verbose=0).items()}
            for callback in callbacks:
                callback(self, epoch, logs)
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("dev_")}, epoch + 1)
            self.add_logs("dev", {k[4:]: v for k, v in logs.items() if k.startswith("dev_")}, epoch + 1)
            verbose and print(epoch_message, "{:.1f}s".format(self._time() - start),
                              *[f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def train_step(self, xs, y):
        """An overridable method performing a single training step.

        A dictionary with the loss and metrics should be returned."""
        self.zero_grad()

        #print(f"Batch size: {xs[0].size(0)}, Input lengths: {xs[1].size(0)}, Target lengths: {y[1].size(0)}")

        y_pred = self.forward(*xs)
        loss = self.compute_loss(y_pred, y, *xs)
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.schedule is not None and self.schedule.step()
            self.loss_metric.update(loss)
            return {"loss": self.loss_metric.compute()} \
                | ({"lr": self.schedule.get_last_lr()[0]} if self.schedule else {}) \
                #| self.compute_metrics(y_pred, y, *xs, training=True)

    def compute_loss(self, y_pred, y, *xs):
        """Compute the loss of the model given the inputs, predictions, and target outputs."""
        return self.loss(y_pred, y)

    def compute_metrics(self, y_pred, y, *xs, training):
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        self.metrics.update(y_pred, y)
        return self.metrics.compute()

    def evaluate(self, dataloader, verbose=1):
        """An evaluation of the model on the given dataset.

        - `dataloader` is the dataset to evaluate on, each element a pair of inputs
          and an output, the inputs either a single tensor or a tuple of tensors;
        - `verbose` controls the verbosity: 0 for silent, 1 for a single message."""
        self.eval()
        self.loss_metric.reset()
        self.metrics.reset()
        for xs, y in dataloader:
            xs, y = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(self.device)
            logs = self.test_step(xs, y)
        verbose and print("Evaluation", *[f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def test_step(self, xs, y):
        """An overridable method performing a single evaluation step.

        A dictionary with the loss and metrics should be returned."""
        with torch.no_grad():
            y_pred = self.forward(*xs)
            self.loss_metric.update(self.compute_loss(y_pred, y, *xs))
            return {"loss": self.loss_metric.compute()}# | self.compute_metrics(y_pred, y, *xs, training=False)

    def predict(self, dataloader, as_numpy=True):
        """Compute predictions for the given dataset.

        - `dataloader` is the dataset to predict on, each element either
          directly the input or a tuple whose first element is the input;
          the input can be either a single tensor or a tuple of tensors;
        - `as_numpy` is a flag controlling whether the output should be
          converted to a numpy array or kept as a PyTorch tensor.

        The method returns a Python list whose elements are predictions
        of the individual examples. Note that if the input was padded, so
        will be the predictions, which will then need to be trimmed."""
        self.eval()
        predictions = []
        for batch in dataloader:
            xs = batch[0] if isinstance(batch, tuple) else batch
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            predictions.extend(self.predict_step(xs, as_numpy=as_numpy))
        return predictions

    def predict_step(self, xs, as_numpy=True):
        """An overridable method performing a single prediction step."""
        with torch.no_grad():
            batch = self.forward(*xs)

            batch = batch.contiguous()

            return batch.numpy(force=True) if as_numpy else batch

    def writer(self, writer):
        """Possibly create and return a TensorBoard writer for the given name."""
        if writer not in self._writers:
            self._writers[writer] = self._SummaryWriter(os.path.join(self.logdir, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        """Log the given dictionary to TensorBoard with a given name and step number."""
        if logs and self.logdir:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    @staticmethod
    def keras_init(module):
        """Initialize weights using the Keras defaults."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                               torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.uniform_(module.weight, -0.05, 0.05)
        if isinstance(module, (torch.nn.RNNBase, torch.nn.RNNCellBase)):
            for name, parameter in module.named_parameters():
                "weight_ih" in name and torch.nn.init.xavier_uniform_(parameter)
                "weight_hh" in name and torch.nn.init.orthogonal_(parameter)
                "bias" in name and torch.nn.init.zeros_(parameter)
                if "bias" in name and isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
                    parameter.data[module.hidden_size:module.hidden_size * 2] = 1


class Model(TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None: # word_dict, lexicon, token_dict) -> None:
        super().__init__()
        # TODO: Define the model.

        self.lstm = torch.nn.LSTM(
            input_size=13,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
        )

        self._output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, len(CommonVoiceCs.LETTERS))
        )

        self.tokens = CommonVoiceCs.LETTERS

        
        self.decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon="lexicon.txt",
            tokens=CommonVoiceCs.LETTERS,#token_dict,
            nbest=3,
            sil_token=" "
        )

        self.apply(self.keras_init)

    def forward(self, mfccs, input_lengths, target_lengths) -> torch.Tensor:
        
        # mfccs: (batch_size, max_sequence_length, num_features)
        # sentence: (batch_size, max_sentence_length)

        # print(input_lengths.shape)

        # print(f"mfccs : {mfccs.shape}")

        # print(f"input_lengths : {input_lengths.shape}")

        packed = torch.nn.utils.rnn.pack_padded_sequence(mfccs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed)

        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        hidden = hidden[:, :, :hidden.size(2) // 2] + hidden[:, :, hidden.size(2) // 2:]
        hidden = self._output_layer(hidden)

        hidden = hidden.permute(1, 0, 2)
        return hidden


    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs, input_lengths, target_lengths) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        return self.loss(y_pred, y_true, input_lengths, target_lengths)

    def ctc_decoding(self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs, input_lengths) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you
        # can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.

        probabilities = torch.nn.functional.softmax(y_pred, dim=2)

        # (batch, frame, num_tokens)

        # probabilities : torch.Size([549, 32, 49])

        # print(f"y_pred : {y_pred.shape}")
        # print(f"probabilities : {probabilities.shape}")

        probabilities = probabilities.permute(1, 0, 2).contiguous().cpu()

        decoded = self.decoder(probabilities)

        #print(f"Decoded : {decoded}")

        return decoded

        # decoded_outputs = self.decoder(probabilities, input_lengths, self.tokens)
        # return decoded_outputs     
        

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, training: bool
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `training==False` to speed up training.
        predictions = None
        self.metrics["edit_distance"].update(predictions, y_true)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            return batch
        

def create_dictionaries(dataset):
    unique_words = set()
    for i in range(len(dataset)):
        sentence = dataset[i]["sentence"].lower()
        unique_words.update(sentence.split())


    word_dict = {word: i + 2 for i, word in enumerate(sorted(unique_words))}  

    lexicon = {word: ' '.join(word) + " " for word in word_dict}

    token_dict = {char: idx for idx, char in enumerate(sorted(set(''.join(unique_words))), start=0)}
    
    return word_dict, lexicon, token_dict


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

    # Load the data.
    common_voice = CommonVoiceCs()

    train_dataset = common_voice.train

    #word_dict, lexicon, token_dict = create_dictionaries(train_dataset)

    def write_lexicon_to_file(lexicon, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for word, characters in lexicon.items():
                file.write(f"{word} {characters.strip()}\n")

    #write_lexicon_to_file(lexicon, 'lexicon.txt')


    def prepare_single_example(example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # Note that `CommonVoiceCs.LETTERS` contains neither padding nor blank symbols, so you
        # need to address it.
        
        # example is of the form: {'mfccs': ..., 'sentence': ...}

        return example['mfccs'], torch.tensor([CommonVoiceCs.LETTERS.index(char) for char in example['sentence']], dtype=torch.int64)
    
    def prepare_single_test(example):
        return example['mfccs']

    train = common_voice.train.transform(prepare_single_example)
    dev = common_voice.dev.transform(prepare_single_example)
    test = common_voice.test.transform(prepare_single_test)
                    
    def prepare_batch(data):
        # TODO: Construct a single batch from a list of individual examples.
        mfccs, sentences = zip(*data)

        input_lengths = torch.tensor([mfcc.shape[0] for mfcc in mfccs], dtype=torch.int64)
        target_lengths = torch.tensor([len(sentence) for sentence in sentences], dtype=torch.int64)

        mfccs_padded = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
        sentence_padded = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)

        return (mfccs_padded, input_lengths, target_lengths), sentence_padded
    
    def prepare_batch_test(data):
        mfccs = data
        input_lengths = torch.tensor([mfcc.shape[0] for mfcc in mfccs], dtype=torch.int64)
        mfccs_padded = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
        return (mfccs_padded, input_lengths, torch.tensor(0)), torch.tensor(0)
        
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, collate_fn=prepare_batch)
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size, collate_fn=prepare_batch_test)

    model = Model(args, train)  #, word_dict, lexicon, token_dict)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CTCLoss(),
        #metrics={"edit_distance": torchmetrics.functional.edit_distance},
        #metrics={"edit_distance": CommonVoiceCs.EditDistanceMetric(0)},
        logdir=args.logdir
    )

    model.fit(train, args.epochs, dev=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test, as_numpy=False)
        """ 
        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS[char] for char in sentence), file=predictions_file)
            return
        """

        for hypothesis in predictions:
            best_hypothesis = hypothesis[0]
            sentence = "".join(CommonVoiceCs.LETTERS[char] for char in best_hypothesis.tokens.tolist())
            print(sentence, file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

