#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
from torchvision import transforms as v2
import torchmetrics
import torchaudio
import torchaudio.models.decoder

import torch.nn.functional as F

from homr_dataset import HOMRDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--visualize", default=False, type=str, help="Visualize the data.")



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
        logs = {}
        for epoch in range(epochs):
            self.train()
            self.loss_metric.reset()
            self.metrics.reset()
            start = self._time()
            epoch_message = f"Epoch={epoch+1}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if verbose == 2 else not verbose)
            for xs, y in data_and_progress:
                #print(f"xs {xs}")
                #print(f"y {y}")
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



class HomrModel(TrainableModule):
    def __init__(self, homr: HOMRDataset, args: argparse.Namespace):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        image_height = 135
        self.dense = torch.nn.Sequential(
            torch.nn.Linear((image_height // 4) * 64, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        self.tokens = homr.MARKS

        self.rnn = torch.nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.output = torch.nn.Linear(128, len(self.tokens) + 1)

        self.decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=None,
            tokens=self.tokens,
            beam_size=3,
            sil_token=" "
        )

    def forward(self, images, input_lengths, mark_lengths):
        images = images.permute(0, 3, 1, 2)
        #print(images.shape)
        x = self.conv(images)
        #print(f"after conv {x.shape}")
        
        batch_size, channels, height, width = x.size()
        x = x.permute(0,3,1,2).reshape(batch_size, width, height * channels)


        x = self.dense(x)
        new_input_lengths = (input_lengths // 4).cpu()

        x = torch.nn.utils.rnn.pack_padded_sequence(x, new_input_lengths, batch_first=True, enforce_sorted=False)

        #x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.output(x)

        return x

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, images, input_lengths, target_lengths) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        #print(f"y_pred {y_pred.shape}")
        #print(f"y_true {y_true.shape}")
        #print(f"input_lengths {input_lengths.shape}")
        #print(f"target_lengths {target_lengths}")

        y_pred = y_pred.permute(1,0,2)
        y_true = y_true.view(-1)
        
        y_true_flat = y_true[y_true != 0]
        true_input_lengths = input_lengths // 4

        return self.loss(y_pred, y_true_flat, true_input_lengths, target_lengths)

    def ctc_decoding(self, y_pred: torch.Tensor, y_true: torch.Tensor, images, input_lengths) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you
        # can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.

        probabilities = torch.nn.functional.softmax(y_pred, dim=2)

        probabilities = probabilities.permute(1, 0, 2).contiguous().cpu()

        decoded = self.decoder(probabilities)

        return decoded

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            #batch = [torch.tensor(x) for x in batch]
            
            if as_numpy:
                batch = [example.numpy(force=True) if isinstance(example, torch.Tensor) else example for example in batch]
                #batch = [example.numpy(force=True) for example in batch]
            return batch

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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[HEIGHT, WIDTH, 1]` tensor of `torch.uint8` values in [0-255] range,
    # - "marks", a `[num_marks]` tensor with indices of marks on the image.
    homr = HOMRDataset(decode_on_demand=True)

    def pad_image(image, target_size=(135, 1400)):
        # Extract current image dimensions
        current_size = image.shape[:2]
       
        #print(f"image shape in pad {image.shape}")

        # Calculate padding for each dimension
        pad_h = target_size[0] - current_size[0]
        pad_w = target_size[1] - current_size[1]
        
        # Padding for height and width
        #padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        
        padding = (0, 0, 0, pad_w, 0, pad_h)

        # Apply padding
        padded_image = F.pad(image, padding, mode='constant', value=-1)  # assuming white padding
        
        #return torch.tensor(padded_image)
        return padded_image

    def prepare_data(example, is_test=False):
        #IMAGE [HEIGHT, WIDTH, 1]
        images = example["image"] / 255.0
        
        #images = [pad_image(img) for img in images] 
        images = pad_image(images)
        #print(f"images len {len(images)}")        
        if is_test:
            return images
        
        return images, example["marks"]




    train = homr.train.transform(lambda example: prepare_data(example, is_test=False))
    dev = homr.dev.transform(lambda example: prepare_data(example, is_test=False))
    test = homr.test.transform(lambda example: prepare_data(example, is_test=True))

    def prepare_batch(data, is_test=False):
        
        if is_test:
            images = torch.stack([torch.tensor(img) for img in data]) 
            images_lengths = torch.tensor([len(image) for image in images])
            images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)
            return (images, images_lengths, torch.tensor(0)), torch.tensor(0)

        images, marks = zip(*data)
        
        #for im in images:
        #    print(im.shape)

        images_lengths = torch.tensor([len(image) for image in images])
        images = torch.stack([torch.tensor(img) for img in images])
        
        marks_lengths = torch.tensor([len(mark) for mark in marks])
        marks = torch.nn.utils.rnn.pad_sequence(marks, batch_first=True)
        marks = torch.stack([x for x in marks])
        

        return (images, images_lengths, marks_lengths), marks

    train = torch.utils.data.DataLoader(train, args.batch_size, shuffle=True, collate_fn=lambda x: prepare_batch(x, is_test=False))
    dev = torch.utils.data.DataLoader(dev, args.batch_size, shuffle=False, collate_fn=lambda x: prepare_batch(x, is_test=False))
    test = torch.utils.data.DataLoader(test, args.batch_size, shuffle=False, collate_fn=lambda x: prepare_batch(x, is_test=True))

    # TODO: Create the model and train it
    model = HomrModel(homr, args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CTCLoss(zero_infinity=True),
        #metrics={"edit_distance": torchmetrics.functional.edit_distance},
        #metrics={"edit_distance": CommonVoiceCs.EditDistanceMetric(0)},
        logdir=args.logdir
    )

    model.fit(train, args.epochs, dev=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions = model.predict(test)

        for sequence in predictions:
            best_hypothesis = sequence[0]
            #print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)
            sentence = "".join(homr.MARKS[char] for char in best_hypothesis.tokens.tolist())
            print(sentence, file=predictions_file)


if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))

