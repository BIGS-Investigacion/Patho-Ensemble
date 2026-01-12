import argparse
import os

import pandas as pd

from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import ExperimentFactory

from create_val_splits import create_validation_splits

def abmil(foundational_model, latent_dim, work_dir, train_source, tissue_patching, task_name, epochs, mtr, create_val, n_folds, fold):
    path_to_split, path_to_task_config = SplitFactory.from_hf(work_dir, train_source, task_name)

    if create_val:
        path_to_original = os.path.join(os.path.dirname(path_to_split), 'k=original.tsv')
        if not os.path.exists(path_to_original):
            pd.read_csv(path_to_split, sep='\t').to_csv(path_to_original, sep='\t', index=False)
        create_validation_splits(path_to_original, path_to_split, n_folds=n_folds)

    if fold != 'all':
      new_path_to_split = os.path.join(os.path.dirname(path_to_split), f'k={fold}.tsv')
      if not os.path.exists(new_path_to_split):
        pd.read_csv(path_to_split, sep='\t', usecols=[0, 1, 2, 3+int(fold)]).to_csv(new_path_to_split, sep='\t', index=False)
      path_to_split = new_path_to_split

    experiment = ExperimentFactory.finetune(
                    split = path_to_split,
                    task_config = path_to_task_config,
                    patch_embeddings_dirs = f'{work_dir}/{train_source}/{tissue_patching}/features_{foundational_model}/',
                    saveto = f'{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}',
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
    experiment.train()
    experiment.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training an abmil model from scratch")
    parser.add_argument("--foundational_model", type=str)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--train_source", type=str)
    parser.add_argument("--tissue_patching", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mtr", type=str)
    parser.add_argument("--create_val", action='store_true', default=False)
    parser.add_argument("--n_folds", type=int, default=50)
    parser.add_argument("--fold", type=str)
    args = parser.parse_args()

    abmil(**vars(args))
