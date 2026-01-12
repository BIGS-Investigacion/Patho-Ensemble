import argparse
import os

import pandas as pd

from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import ExperimentFactory

import shutil

def abmil(foundational_model, latent_dim, work_dir, train_source, tissue_patching, task_name, epochs):

    if not os.path.exists(f'{work_dir}/{train_source}/{task_name}/k=train.tsv'):
        df = pd.read_csv(f'{work_dir}/{train_source}/{task_name}/k=all.tsv', sep="\t")
        df = df.replace({"train": "__tmp__", "val": "train"})
        df = df.replace({"__tmp__": "val"})
        df.to_csv(f'{work_dir}/{train_source}/{task_name}/k=train.tsv', sep="\t", index=False)

    if not os.path.exists(f'{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}_train_eval'):
        shutil.copytree(f'{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}', f'{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}_train_eval')

    experiment = ExperimentFactory.finetune(
                    split = f'{work_dir}/{train_source}/{task_name}/k=train.tsv',
                    task_config = f'{work_dir}/{train_source}/{task_name}/config.yaml',
                    patch_embeddings_dirs = f'{work_dir}/{train_source}/{tissue_patching}/features_{foundational_model}/',
                    saveto = f'{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}_train_eval',
                    combine_slides_per_patient = False,
                    model_name = 'abmil',
                    bag_size = 2048,
                    base_learning_rate = 0.0002,
                    gradient_accumulation = 1,
                    weight_decay = 0.00001,
                    num_epochs = epochs,
                    scheduler_type = 'cosine',
                    optimizer_type = 'AdamW',
                    balanced = True,
                    save_which_checkpoints = 'best-val-loss',
                    model_kwargs = {
                        'input_feature_dim': latent_dim,
                        'n_heads': 1,
                        'head_dim': 512,
                        'dropout': 0.25,
                        'gated': False
                        }
                    )
    experiment.validate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for eval training set")
    parser.add_argument("--foundational_model", type=str)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--train_source", type=str)
    parser.add_argument("--tissue_patching", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    abmil(**vars(args))
