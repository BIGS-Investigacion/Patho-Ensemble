import os
from tqdm import tqdm
import json
import pandas as pd
import h5py
from contextlib import ExitStack
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from optuna import create_study
import optuna
import pygad
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.distributions import FloatDistribution

from patho_bench.experiments.utils.ClassificationMixin import ClassificationMixin
from patho_bench.experiments.BaseExperiment import BaseExperiment


class Metrics(ClassificationMixin, BaseExperiment):
    def __init__(self,
                 task_type: str,
                 model_kwargs: dict,
                 num_bootstraps: int,
                 results_dir: str,
                 split: str,
                 num_folds: int,
                 all_labels_across_folds: list,
                 all_preds_across_folds: list):
        """
        Base class for all experiments.

        Args:
            task_type (str): Type of task. Can be 'classification' or 'survival'.
            dataset (BaseDataset): Dataset object
            batch_size (int): Batch size.
            model_constructor (callable): Model class which can be called to create model instance.
            model_kwargs: Arguments passed to model_constructor.
            num_epochs (int): Number of epochs.
            accumulation_steps (int): Number of batches to accumulate gradients over before stepping optimizer.
            optimizer_config: Optimizer config.
            scheduler_config: LR scheduler config.
            save_which_checkpoints (str): Mode of saving checkpoints.
            num_bootstraps (int): Number of bootstraps to use for computing 95% CI.
            precision (torch.dtype): Precision to use for training.
            device (str): Device to use for training.
            results_dir (str): Where to save results.
            view_progress (str, optional): How to log progress. Can be 'bar' or 'verbose'. Defaults to 'bar'.
            lr_logging_interval (int, optional): Interval at which to log learning rate to dashboard (in number of accumulation steps). Defaults to None (do not log).
            seed (int): Seed for reproducibility.
            **kwargs: Additional arguments to save in config.json
        """
        self.task_type = task_type
        self.model_kwargs = model_kwargs
        self.num_bootstraps = num_bootstraps
        self.results_dir = results_dir
        self.split = split
        self.num_folds = num_folds
        self.all_labels_across_folds = all_labels_across_folds
        self.all_preds_across_folds = all_preds_across_folds

    def run(self):
        labels = []
        preds = []
        scores = []

        if self.num_folds == 1:
            # If only one fold or one sample per fold, will save results at end across all folds
            labels = self.all_labels_across_folds
            preds = self.all_preds_across_folds
        else:
            # If multiple folds and multiple samples per fold, save per-fold results
            for f in range(self.num_folds):
                per_fold_save_dir = os.path.join(self.results_dir, f'{self.split}_metrics', f'fold_{f}')
                scores.append(self._compute_metrics(self.all_labels_across_folds[f], self.all_preds_across_folds[f], per_fold_save_dir))

        # After collecting all folds, either do bootstrapping or an average across folds
        summary = self._finalize_metrics(self.split, labels, preds, scores)

        with open(os.path.join(self.results_dir, f'{self.split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

    def _compute_metrics(self, labels, preds, save_dir):
        """
        Save metrics to file and return a dictionary of metrics.

        Args:
            labels (np.array or dict): Ground truth labels
            preds (np.array): Predictions
            save_dir (str): Directory to save metrics to
        """
        if self.task_type == 'classification':
            self.auc_roc(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "roc_curves.png"))
            self.confusion_matrix(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "confusion_matrices.png"))
            self.precision_recall(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "pr_curves.png"))
            scores = self.classification_metrics(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "metrics.json"))
            return scores['overall']

    def _finalize_metrics(self, split, labels_across_folds, preds_across_folds, scores_across_folds):
        """
        Combine per-fold results or do bootstrapping if single fold

        Arguments:
            split (str): Split name ('val' or 'test')
            labels_across_folds (list): List of labels across folds
            preds_across_folds (list): List of predictions across folds
            scores_across_folds (list): List of scores across folds

        Returns:
            summary (dict): Dictionary of summary metrics
        """
        if len(labels_across_folds) > 0:
            # Perform bootstrapping and calculate 95% CI
            bootstraps = self.bootstrap(labels_across_folds, preds_across_folds, self.num_bootstraps)
            if self.task_type == 'classification':
                scores_across_folds = [self.classification_metrics(labels, preds, self.model_kwargs['num_classes'])['overall'] for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')]

            # Save bootstraps
            folder_path = os.path.join(self.results_dir, f"{split}_metrics")
            os.makedirs(folder_path, exist_ok=True)
            for idx, metrics_dict in enumerate(scores_across_folds):
                folder_path_curr = os.path.join(folder_path, f"bootstrap_{idx}")
                os.makedirs(folder_path_curr, exist_ok=True)

                file_path = os.path.join(folder_path_curr, "metrics.json")
                with open(file_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

            return self.get_95_ci(scores_across_folds)
        else:
            # Report mean ± SE across folds
            return self.get_mean_se(scores_across_folds)


class MetricDistance:
    def __init__(self,
                 foundational_models: list,
                 work_dir: str,
                 train_source: str,
                 tissue_patching: str,
                 task_name: str,
                 metric: str,
                 fold: int):
        self.metric = metric
        self.fold = fold

        self.dirs = [os.path.join(work_dir, train_source, task_name, "abmil", f'{f}_{tissue_patching}', 'val_metrics') for f in foundational_models]

        for d in self.dirs:
            assert os.path.exists(d), f"Missing path: {d}"

    def run(self):
        fold_values = []
        for dir in self.dirs:
            metrics_path = os.path.join(dir, f"fold_{self.fold}/metrics.json")
            if not os.path.exists(metrics_path):
                metrics_path = os.path.join(dir, f"bootstrap_{self.fold}/metrics.json")
                if not os.path.exists(metrics_path):
                    raise FileNotFoundError(f"{dir}/... does not exist")
                metrics_path = os.path.join(os.path.dirname(dir), "val_metrics_summary.json")
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                fold_values.append(metrics[self.metric]["mean"])
            else:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                fold_values.append(metrics["overall"][self.metric])

        fold_values = np.array(fold_values, dtype=float)

        if fold_values.sum() == 0:
            weights = np.ones_like(fold_values) / len(fold_values)
        else:
            weights = fold_values / fold_values.sum()

        return np.array(weights)
