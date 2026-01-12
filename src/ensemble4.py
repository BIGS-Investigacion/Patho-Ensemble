import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from utils import Metrics, MetricDistance


class Ensemble(nn.Module):
    def __init__(self, xdirs, vdirs, tdirs, meta_model, device="cpu"):
        super().__init__()

        self.xdirs = xdirs
        self.vdirs = vdirs
        self.tdirs = tdirs
        self.meta_model = meta_model
        self.device = device

    def forward(self):
        xlabels, xpreds = [], []
        for d in self.xdirs:
            xlabels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            xpreds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        for i in range(1, len(xlabels)):
            assert torch.equal(xlabels[0], xlabels[i]), f"Labels difieren entre modelo 0 y {i}"

        vlabels, vpreds = [], []
        for d in self.vdirs:
            vlabels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            vpreds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        for i in range(1, len(vlabels)):
            assert torch.equal(vlabels[0], vlabels[i]), f"Labels difieren entre modelo 0 y {i}"

        xstacked = torch.stack(xpreds, dim=0).to(self.device)
        vstacked = torch.stack(vpreds, dim=0).to(self.device)

        xlabels = xlabels[0]
        vlabels = vlabels[0]

        X_train = xstacked.permute(1, 0, 2).reshape(xstacked.shape[1], -1).cpu().numpy()
        y_train = xlabels.cpu().numpy()

        X_val = vstacked.permute(1, 0, 2).reshape(vstacked.shape[1], -1).cpu().numpy()
        y_val = vlabels.cpu().numpy()

        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)

        self.meta_model.fit(X_train, y_train)

        base_path = Path(self.xdirs[0]).parent.parent.parent / "ensemble4"
        base_path.mkdir(parents=True, exist_ok=True)

        coefs_path = base_path / "coefs.npy"
        odds_path = base_path / "odds_ratios.npy"

        coefs = np.array(self.meta_model.coef_[0])
        odds = np.exp(coefs)

        if coefs_path.exists():
            old = np.load(coefs_path)
            new = np.vstack([old, coefs])
        else:
            new = np.array([coefs])

        np.save(coefs_path, new)

        if odds_path.exists():
            old = np.load(odds_path)
            new = np.vstack([old, odds])
        else:
            new = np.array([odds])

        np.save(odds_path, new)

        tlabels, tpreds = [], []
        for d in self.tdirs:
            tlabels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            tpreds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        tstacked = torch.stack(tpreds, dim=0).to(self.device)
        num_models, N_test, num_classes = tstacked.shape
        X_test = tstacked.permute(1, 0, 2).reshape(N_test, num_models * num_classes).cpu().numpy()

        probs = self.meta_model.predict_proba(X_test)
        final_probs = torch.from_numpy(probs).to(self.device).float()

        return final_probs, tlabels[0]


def main(foundational_models, work_dir, train_source, tissue_patching, task_name):
    dirs = [os.path.join(work_dir, train_source, task_name, 'abmil', f'{f}_{tissue_patching}') for f in foundational_models]
    xdirs = []
    vdirs = []
    tdirs = []

    folds = []
    for idx, d in enumerate(dirs):
        if any(name.startswith("fold_") for name in os.listdir(os.path.join(d, 'val_outputs'))):
            folds.append(len(os.listdir(os.path.join(d, 'val_outputs'))))
        else:
            folds.append(1)
        xdirs.append(os.path.join(f'{d}_train_eval', 'val_outputs'))
        vdirs.append(os.path.join(d, 'val_outputs'))
        tdirs.append(os.path.join(d, 'test_outputs'))

    assert len(set(folds)) == 1, "Mismatch in number of folds"

    n_folds = folds[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_preds = []
    all_labels = []

    print("\n\nEvaluating ensemble...")
    with tqdm(total=n_folds, unit="fold") as pbar:
        for f in range(n_folds):
            pbar.set_description(f"Evaluating fold {f+1}/{n_folds}")

            preds, labels = Ensemble(
                [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in xdirs],
                [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in vdirs],
                [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in tdirs],
                LogisticRegression(max_iter=1000),
                device=device,
            )()

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pbar.update(1)

    tqdm.write("🔎 Running metrics across all folds...")
    Metrics(task_type='classification',
            model_kwargs={'num_classes': len(set(all_labels[0]))},
            num_bootstraps=100,
            results_dir=os.path.join(work_dir, train_source, task_name, 'abmil', f'ensemble4'),
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
    args = parser.parse_args()

    main(**vars(args))
