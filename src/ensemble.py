import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse

from utils import Metrics, MetricDistance


class Ensemble(nn.Module):
    def __init__(self, dirs, weights=None, random_init=False, device="cpu"):
        super().__init__()

        assert len(dirs) > 0, "Need at least one model directory"
        for d in dirs:
            assert os.path.exists(d), f"Missing path: {d}"

        self.dirs = dirs
        self.device = device

        if random_init:
            w = torch.randn(len(dirs), dtype=torch.float32)
        else:
            assert weights is not None, "Must provide weights if random_init=False"
            assert len(dirs) == len(weights), "Mismatch in dirs and weights length"
            w = torch.tensor(weights, dtype=torch.float32)

        self.weights = nn.Parameter(w, requires_grad=True)

    def forward(self):
        labels, preds = [], []

        for d in self.dirs:
            labels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            preds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        for i in range(1, len(labels)):
            assert torch.equal(labels[0], labels[i]), f"Labels differ between model 0 and model {i} in fold {fold}"

        stacked = torch.stack(preds, dim=0).to(self.device)
        norm_w = torch.softmax(self.weights, dim=0).to(self.device)
        weighted_avg = torch.tensordot(norm_w, stacked, dims=([0], [0]))

        return weighted_avg, labels[0]


def main(foundational_models, work_dir, train_source, tissue_patching, task_name, weights_type):
    dirs = [os.path.join(work_dir, train_source, task_name, 'abmil', f'{f}_{tissue_patching}') for f in foundational_models]

    folds = []
    for idx, d in enumerate(dirs):
        path = os.path.join(d, 'test_outputs')
        assert os.path.exists(path), f"Missing path: {path}"
        has_fold = any(name.startswith("fold_") for name in os.listdir(path))
        if has_fold:
            folds.append(len(os.listdir(path)))
        else:
            folds.append(1)
        dirs[idx] = path

    assert len(set(folds)) == 1, "Mismatch in number of folds"

    n_folds = folds[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'weights.npy')
    if os.path.exists(weights_path):
        weights = np.load(weights_path)
    else:
        weights = []
        print("\n\nComputing weights...")
        with tqdm(total=n_folds, unit="fold") as pbar:
            for f in range(n_folds):
                pbar.set_description(f"Processing fold {f+1}/{n_folds}")
                w = MetricDistance(foundational_models, work_dir, train_source, tissue_patching, task_name, weights_type, f).run()
                weights.append(w)
                pbar.update(1)
            weights = np.array(weights)
            np.save(weights_path, weights)
            tqdm.write(f"✅ All folds complete. Weights saved to {weights_path}")

    all_preds = []
    all_labels = []

    print("\n\nEvaluating ensemble...")
    with tqdm(total=n_folds, unit="fold") as pbar:
        for f in range(n_folds):
            pbar.set_description(f"Evaluating fold {f+1}/{n_folds}")

            preds, labels = Ensemble(
                [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in dirs],
                weights=weights[f],
                random_init=False,
                device=device,
            )()

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pbar.update(1)

    tqdm.write("🔎 Running metrics across all folds...")
    Metrics(task_type='classification',
            model_kwargs={'num_classes': len(set(all_labels[0]))},
            num_bootstraps=100,
            results_dir=os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble'),
            split='test',
            num_folds=n_folds,
            all_labels_across_folds=all_labels,
            all_preds_across_folds=all_preds).run()
    tqdm.write("✅ Metrics computation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluate ensemble")
    parser.add_argument("--foundational_models", nargs='+', type=str)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--train_source", type=str)
    parser.add_argument("--tissue_patching", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--weights_type", type=str)
    args = parser.parse_args()

    main(**vars(args))
