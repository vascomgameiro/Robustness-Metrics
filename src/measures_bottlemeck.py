import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss
from skorch.callbacks import Callback


def clone_trainer(trainer, is_reinit_besides_param=False):
    """Clone a trainer with optional possibility of reinitializing everything besides
    parameters (e.g. optimizers.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer_new = copy.deepcopy(trainer)

    if is_reinit_besides_param:
        trainer_new.initialize_callbacks()
        trainer_new.initialize_criterion()
        trainer_new.initialize_optimizer()
        trainer_new.initialize_history()

    return trainer_new


class NegCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        # select label from the targets
        return -super().forward(input, target[0])


class OnlineVariance:
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


class StoreVarGrad(Callback):
    """Callback which applies a function on all gradients, stores the variance during each epoch."""

    def __init__(self):
        self.online_vars = dict()
        self.n = 0
        self.var_grads = dict()

    def initialize(self):
        self.online_vars = dict()
        self.n = 0
        self.var_grads = dict()

    def on_grad_computed(self, net, **kwargs):
        for name, param in net.module_.named_parameters():
            if param.grad is not None:
                if name not in self.online_vars:
                    self.online_vars[name] = OnlineVariance()

                self.online_vars[name].include(param.grad.cpu().detach().flatten().numpy())

    def on_epoch_end(self, net, parent=None, **kwargs):
        epoch = net.history[-1]["epoch"]
        self.n += 1

        self.var_grads = {k: v.variance for k, v in self.online_vars.items()}
        self.online_vars = dict()


def eval_clf(trainer, dataset):
    """
    Evaluate a classifier on a dataset.
    Returns accuracy, top-5 accuracy, and log-likelihood.
    """
    y_pred_proba = trainer.predict_proba(dataset)
    loglike = -log_loss(dataset.targets, y_pred_proba)
    y_pred = y_pred_proba.argmax(-1)
    accuracy = accuracy_score(dataset.targets, y_pred)
    top5_acc = top_n_accuracy_score(dataset.targets, y_pred_proba, n=5)

    return {"accuracy": accuracy, "top5_acc": top5_acc, "loglike": loglike}


def top_n_accuracy_score(y_true, y_pred, n=5, normalize=True):
    """
    Compute top-N accuracy for multiclass classification.
    """
    num_obs, num_labels = y_pred.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1 :]:
            counter += 1
    return counter / num_obs if normalize else counter


def get_path_norm(trainer, dataset):
    """
    Compute the path norm for a classifier, measuring model capacity.
    """
    trainer = clone_trainer(trainer)
    trainer.module_.transformer.is_use_mean = True

    with torch.no_grad():
        for _, W in trainer.module_.named_parameters():
            W.pow_(2)

    all_ones = dataset[0][0].unsqueeze(0).fill_(1)
    logits = trainer.forward(all_ones)[0]
    sum_logits = logits.sum().item()
    return sum_logits**0.5


def get_var_grad(trainer, dataset):
    """
    Compute the variance of gradients for model parameters.
    """
    trainer = clone_trainer(trainer)
    trainer.module_.is_freeze_transformer = False
    trainer.callbacks_.append(("store_grad", StoreVarGrad()))

    trainer.check_data(dataset, None)
    trainer.notify("on_epoch_begin", dataset_train=dataset, dataset_valid=dataset)
    trainer._single_epoch(dataset, training=True, epoch=0)

    for _, cb in trainer.callbacks_:
        if isinstance(cb, StoreVarGrad):
            cb.on_epoch_end(trainer, dataset_train=dataset, dataset_valid=dataset)
            var_grad = np.concatenate([v.flatten() for v in cb.var_grads.values()])

    return var_grad.mean()


def get_sharp_mag(trainer, dataset, target_deviation=0.1, max_binary_search=50, sigma_min=0, sigma_max=2):
    """
    Compute sharpness magnitude, reflecting sensitivity to parameter perturbations.
    """
    trainer = clone_trainer(trainer)
    acc = accuracy_score(dataset.targets, trainer.predict(dataset))
    trainer.criterion = NegCrossEntropyLoss()

    for _ in range(max_binary_search):
        sigma_min, sigma_max = get_sharp_mag_interval(
            trainer,
            acc,
            dataset,
            sigma_min,
            sigma_max,
            target_deviation,
            n_restart_perturbate,
            is_relative,
        )

        if sigma_min > sigma_max or math.isclose(sigma_min, sigma_max, rel_tol=1e-2):
            break

    return 1 / (sigma_max**2)


def get_H_Q_xCz(trainer, dataset, select="H_Q_xCz", Q_zx=None, max_epochs=100, batch_size=256, lr=1e-2):
    """
    Compute information-theoretic measures like H_Q[X|Z].
    """
    trainer = clone_trainer(trainer)
    model = trainer.module_.transformer
    model.is_transform = False  # Ensure deterministic behaviour

    dib = DIBLoss(
        Q_zx if Q_zx else copy.deepcopy(trainer.module_.clf),
        n_classes=dataset.n_classes,
        z_dim=model.z_dim,
        map_target_position=dataset.map_target_position,
    )

    net = NeuralNetTransformer(
        module=model,
        criterion=dib,
        optimizer=torch.optim.Adam,
        lr=lr,
        max_epochs=max_epochs,
        train_split=None,
        batch_size=batch_size,
        device=trainer.device,
    )

    net.fit(dataset, y=None)
    results = eval_clf(net, dataset)
    return results[select]
    results = eval_clf(net, dataset)
    return results[select]
    return results[select]
    return results[select]
    return results[select]
