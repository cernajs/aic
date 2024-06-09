#!/usr/bin/env python3

# 21e65bb3-23db-11ec-986f-f39926f24a9c
# e4553c7b-e907-46c6-98e5-f08d1ce8f040

import argparse
import datetime
import os
import re

import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
import torchmetrics
import torch.nn.functional as F
import torchvision.transforms as transforms

from homr_dataset import HOMRDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=512, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
parser.add_argument("--rnn_layers", default=3, type=int, help="RNN dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


homr_bow = HOMRDataset.MARKS.index("bow")
homr_eow = HOMRDataset.MARKS.index("eow")

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
        y_pred = self.forward(*xs)
        loss = self.compute_loss(y_pred, y, *xs)
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.schedule is not None and self.schedule.step()
            self.loss_metric.update(loss)
            return {"loss": self.loss_metric.compute()} \
                | ({"lr": self.schedule.get_last_lr()[0]} if self.schedule else {}) \
                | self.compute_metrics(y_pred, y, *xs, training=True)

    def compute_loss(self, y_pred, y, *xs):
        """Compute the loss of the model given the inputs, predictions, and target outputs."""
        #print(f"y_pred {y_pred.shape}")
        #print(f"y {y.shape}")
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
            return {"loss": self.loss_metric.compute()} | self.compute_metrics(y_pred, y, *xs, training=False)

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


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention vo the given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # TODO: Define
        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output

        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)

        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)

        self._output_layer = torch.nn.Linear(attention_dim, 1)

    def setup_memory(self, encoded):
        self._encoded = encoded
        # TODO: Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        # TODO: Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        # - Sum the two projections. However, the first has shape `[batch_size, input_sequence_len, attention_dim]`
        #   and the second just `[batch_size, attention_dim]`, so the second needs to be expanded so that
        #   the sum of the projections broadcasts correctly.
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        # - Then, run softmax on a suitable axis, generating `weights`.
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input form had.
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.

        projection = self._encoded_projected

        projected_decoder = self._project_decoder_layer(states)

        sum = projection + projected_decoder.unsqueeze(1)

        output = self._output_layer(torch.tanh(sum))

        probs = torch.nn.functional.softmax(output, dim=1)

        attention = torch.sum(self._encoded * probs, dim=1)

        #print(f"attention {attention.shape}")
        #print(f"inputs {inputs.shape}")

        stacked = torch.cat([inputs, attention], dim=1)

        #print(f"stacked {stacked.shape}")

        ret = self._cell(stacked, states)
        #print(f"return shape of attention {ret.shape}")
        return ret


class Model(TrainableModule):
    def __init__(self, homr: HOMRDataset, args: argparse.Namespace) -> None:
        super().__init__()
        self._target_vocab = homr.MARKS
        self.args = args

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars

        #self._source_rnn = torch.nn.GRU(args.cle_dim, args.rnn_dim, bidirectional=True, batch_first=True)

        # TODO: Define
        # - `self._target_rnn_cell` as a `WithAttention` with `attention_dim=args.rnn_dim`, employing as the
        #   underlying cell the `torch.nn.GRUCell` with `args.rnn_dim`. The cell will process concatenated
        #   target character embeddings and the result of the attention mechanism.
        # self._target_rnn_cell = WithAttention(torch.nn.GRUCell(args.rnn_dim + 64, args.rnn_dim), args.rnn_dim)

        self._target_rnn_cell = WithAttention(torch.nn.GRUCell(args.rnn_dim + args.cle_dim, args.rnn_dim), args.rnn_dim)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        
        """
        image_height = 200
        image_width = 1720
        self.dense = torch.nn.Sequential(
            #torch.nn.Linear((image_width // 16) * (image_height // 16) * 256, 512),
            torch.nn.Linear(256,512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        self.rnn = torch.nn.LSTM(
                        input_size=256 * 12,
                        hidden_size=args.rnn_dim,
                        num_layers=2,
                        bidirectional=True,
                        batch_first=True,
                        dropout=0.2
                    )
        
        self.secrnn = torch.nn.LSTM(
                        input_size=args.rnn_dim,
                        hidden_size=args.rnn_dim,
                        num_layers=2,
                        bidirectional=True,
                        batch_first=True,
                        dropout=0.2
                    )
        """

        self._dropout_layer = torch.nn.Dropout(args.dropout)

        self._rnn = torch.nn.ModuleList([torch.nn.LSTM(256 * 12, args.rnn_dim, batch_first=True, bidirectional=True)])
        self._rnn.extend([torch.nn.LSTM(args.rnn_dim, args.rnn_dim, batch_first=True, bidirectional=True) for _ in range(args.rnn_layers - 1)])

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_output_layer = torch.nn.Linear(args.rnn_dim, len(self._target_vocab))

        if not args.tie_embeddings:
            # TODO(lemmatizer_noattn): Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            
            #self._target_embedding = torch.nn.Embedding(len(self._target_vocab), 32)
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab), args.cle_dim)
        else:
            # TODO(lemmatizer_noattn): Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            def embedding_function(chars):
                return torch.nn.functional.embedding(
                    chars, self._target_output_layer.weight * (args.rnn_dim ** 0.5)
                )

            self._target_embedding = embedding_function
        
        #print(f"len(self._target_vocab) {len(self._target_vocab)}")
        # Initialize the layers using the Keras-inspired initialization. You can try
        # removing this line to see how much worse the default PyTorch initialization is.
        self.apply(self.keras_init)

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0


    def forward(self, images, input_lengths = None, marks = None, mark_lengths = None):
        if input_lengths is None:     
            #only for prints while training
            il = torch.tensor([160, 160])
            encode = self.encoder(images, il)
        
        else:
            encode = self.encoder(images, input_lengths)
        
        if mark_lengths is not None:
            return self.decoder_training(encode, marks)
        else:
            return self.decoder_prediction(encode, max_length=encode.shape[1] + 10)

    def encoder(self, images: torch.Tensor, input_lengths) -> torch.Tensor:

        images = images.permute(0, 3, 1, 2)

        x = self.conv(images)

        batch_size, channels, height, width = x.size()
        x = x.permute(0,3,1,2).reshape(batch_size, width, height * channels)
       
        new_input_lengths = (input_lengths // 16).cpu()
        hidden = torch.nn.utils.rnn.pack_padded_sequence(x, new_input_lengths, batch_first=True, enforce_sorted=False)
       
        for i, rnn in enumerate(self._rnn):
            residual = hidden
            hidden, _ = rnn(hidden)
            forward, backward = torch.chunk(hidden.data, 2, dim=-1)
            hidden = self._dropout_layer(forward + backward)
            if i:
                hidden += residual.data
            hidden = torch.nn.utils.rnn.PackedSequence(hidden, *residual[1:])

        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        return hidden

        """
        x, _ = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)


        forward, backward = torch.chunk(x, 2, dim=-1)
        x = forward + backward
       

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, new_input_lengths, batch_first=True, enforce_sorted=False)
        residual, _ = self.secrnn(x_pack)
        residual, _ = torch.nn.utils.rnn.pad_packed_sequence(residual, batch_first=True)

        forward, backward = torch.chunk(residual, 2, dim=-1)
        hidden = forward + backward

        x = x + hidden

        return x
        """        

    def decoder_training(self,  encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_list = targets[:, :-1].tolist()
        inputs = [[homr_bow] + target for target in targets_list]

        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(inp, dtype=torch.long) for inp in inputs],
            batch_first=True, padding_value=0
        ).to(encoded.device)

        self._target_rnn_cell.setup_memory(encoded)

        x = self._target_embedding(inputs)
        
        state = encoded[:,0,:]
        
        outputs = []
        for i in range(inputs.size(1)):
            x_t = x[:, i, :]
            state = self._target_rnn_cell(x_t, state)
            outputs.append(state)
        
        outputs = torch.stack(outputs, dim=1)
        
        ret = self._target_output_layer(outputs).permute(0, 2, 1)
        return ret

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        # TODO(decoder_training): Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded[:, 0]`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = 0
        inputs = torch.full((batch_size,), homr_bow, dtype=torch.long, device=encoded.device)
        states = encoded[:, 0]
        results = []
        result_lengths = torch.full((batch_size,), max_length, dtype=torch.long, device=encoded.device)

        while index < max_length and torch.any(result_lengths == max_length):
            # TODO(lemmatizer_noattn):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            embeding = self._target_embedding(inputs)

            hidden = self._target_rnn_cell(embeding, states)
            states = hidden

            rnn_output = self._target_output_layer(hidden)

            predictions = rnn_output.argmax(dim=-1)

            # Store the predictions in the `results` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(predictions == homr_eow) & (result_lengths > index)] = index + 1

            # TODO(lemmatizer_noattn): Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs, training):
        if training:
            y_pred = y_pred.argmax(dim=-2)
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=0)
        #self.metrics["accuracy"](torch.all((y_pred == y) | (y == 0), dim=-1))
        return self.metrics.compute()

    def train_step(self, xs, y):
        result = super().train_step(xs, y)
        
        self._batches += 1
       
        if self._batches % self._show_results_every_batch == 0 and False:
            
            predicted_indices = self.predict_step((xs[0][:self.args.batch_size],))[0]
            predicted_indices = predicted_indices.tolist()
            predicted_words = [self._target_vocab[idx] for idx in predicted_indices]

            self._tqdm.write("{}: {} -> {}".format(
                self._batches,
                "idk",#"".join(self._source_vocab.strings(np.trim_zeros(xs[0][0].numpy(force=True)))),
                #"".join(self._target_vocab[self.predict_step((xs[0][:2],))[0]])))
                "".join(predicted_words)))

        return result
    
    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs, training=False)
    
    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            #batch = self.decoder_prediction(*xs)
            # If `as_numpy==True`, trim t.shhe predictions at the first EOW.
            if as_numpy:
                batch = [example[np.cumsum(example == homr_eow) == 0] for example in batch.numpy(force=True)]
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

    """
    def pad_image(image, target_size=(200, 1720)):
        # Extract current image dimensions
        current_size = image.shape[:2]

        target_height = 160
        resize_transform = transforms.Resize((target_height, int(image.shape[1] * (target_height / image.shape[0]))))
        image = resize_transform(image)
        
        #print(f"image shape in pad {image.shape}")

        # Calculate padding for each dimension
        pad_h = target_size[0] - current_size[0]
        pad_w = target_size[1] - current_size[1]

        # Padding for height and width
        #padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)

        #padding = (0, 0, 0, pad_w, 0, pad_h)
        padding = (0, 0, 0, pad_w, 0, 0)

        # Apply padding
        padded_image = F.pad(image, padding, mode='constant', value=-1)  # assuming white padding

        #return torch.tensor(padded_image)
        return padded_image
    """

    def pad_image(image, target_height=200, target_width=1720, padding_value=-1):
        # Resize the height to target_height while maintaining the aspect ratio
        #resize_transform = transforms.Resize((target_height, int(image.shape[1] * (target_height / image.shape[0]))))
        resize_transform = transforms.Resize((target_height, image.shape[1]))
        resized_image = resize_transform(image.permute(2, 0, 1))
        resized_image = resized_image.permute(1, 2, 0)

        #print(f"image {image.shape}")
        #print(f"resized_image {resized_image.shape}")

        current_height, current_width = resized_image.shape[:2]
        pad_w = target_width - current_width
        

        # Apply padding to width
        padding = (0, 0, 0, pad_w)  # (left, right, top, bottom)

        # Apply padding
        padded_image = F.pad(resized_image, padding, mode='constant', value=padding_value)

        return padded_image

    def prepare_data(example, is_test=False):
        #IMAGE [HEIGHT, WIDTH, 1]
        images = example["image"] / 255.0

        images_lengths = torch.tensor([len(image) for image in images])
        height, width, channels = images.shape 
        images = pad_image(images)

        if is_test:
            return images, torch.tensor([height, width])
        
        
        marks = torch.cat((example["marks"], torch.tensor([homr_eow])))

        return images, marks, torch.tensor([height, width])




    train = homr.train.transform(lambda example: prepare_data(example, is_test=False))
    dev = homr.dev.transform(lambda example: prepare_data(example, is_test=False))
    test = homr.test.transform(lambda example: prepare_data(example, is_test=True))

    def prepare_batch(data, is_test=False):

        if is_test:
            image, lengths = zip(*data)
            #print(lengths)
            images = torch.stack([torch.tensor(img) for img in image])
            #images_lengths = torch.tensor([len(image) for image in images])
            images_lengths = torch.tensor([length[1] for length in lengths])
            images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True, padding_value=-1)
            
            #print(images_lengths)

            return (images, images_lengths), torch.tensor(0)

        images, marks, lengths = zip(*data)

        #images_lengths = torch.tensor([len(image) for image in images])
        images_lengths = torch.tensor([length[1] for length in lengths])
        images = torch.stack([torch.tensor(img) for img in images])

        marks_lengths = torch.tensor([len(mark) for mark in marks])
        marks = torch.nn.utils.rnn.pad_sequence(marks, batch_first=True, padding_value=0)
        marks = torch.stack([x for x in marks])


        return (images, images_lengths, marks,  marks_lengths), marks

    train = torch.utils.data.DataLoader(train, args.batch_size, shuffle=True, collate_fn=lambda x: prepare_batch(x, is_test=False))
    dev = torch.utils.data.DataLoader(dev, args.batch_size, shuffle=False, collate_fn=lambda x: prepare_batch(x, is_test=False))
    test = torch.utils.data.DataLoader(test, args.batch_size, shuffle=False, collate_fn=lambda x: prepare_batch(x, is_test=True))

    # TODO: Create the model and train it
    model = Model(homr, args)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    model.configure(
        optimizer=optimizer,
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train)),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(
            "multiclass", num_classes=1000, ignore_index=0)},
        logdir=args.logdir
    )

    model.fit(train, args.epochs, dev=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions = model.predict(test)

        for sequence in predictions:
            print(sequence)
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))

