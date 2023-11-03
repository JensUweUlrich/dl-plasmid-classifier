import click
import os
import glob
import pandas as pd
import time
import torch
import pytorch_lightning as pl

from dataloader import CustomDataLoader
from models.SquiggleNetModel import Bottleneck, SquiggleNetLightning
from models.DeepSelectNetModel import DSBottleneck, DeepSelectNet
from models.BNLSTM import bnLSTM
from MitoModel import VDCNN_bnLSTM_1window, VDCNN_gru_1window_hidden, regGru_32window_hidden_BN
from numpy import random
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor, ModelSummary, ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.train.lightning import RayDDPStrategy, RayFSDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from functools import partial

PLASMID_LABEL = 0
CHR_LABEL = 1


@click.command()
@click.option('--p_train', '-pt', help='folder with plasmid training tensor files', required=True,
              type=click.Path(exists=True))
@click.option('--p_val', '-pv', help='folder with plasmid validation tensor files', required=True,
              type=click.Path(exists=True))
@click.option('--p_ids', '-pid', help='file path of plasmid validation read ids', default=None, required=False)
@click.option('--chr_train', '-ct', help='folder with chromosome training tensor files', required=True)
              #type=click.Path(exists=True))
@click.option('--chr_val', '-cv', help='folder with chromosome validation tensor files', required=True,
              type=click.Path(exists=True))
@click.option('--chr_ids', '-cid', help='file path of chromosome validation read ids', default=None, required=False)
@click.option('--out_folder', '-o', help='output folder path in which logs and models are saved', required=True,
              type=click.Path())
@click.option('--batch_size', '-b', default=1000, help='batch size, default 1000 reads')
@click.option('--n_workers', '-w', default=4, help='number of workers, default 4')
@click.option('--random_seed', '-s', default=42, help='random seed for file shuffling of custom data loaders')
@click.option('--train_model', '-m', default='SquiggleNet', help='deep learning model used for training (SquiggleNet, DeepSelectNet, bnLSTM, bnGRU, vdCNN_GRU, vdCNN_bnLSTM, regGRU)')
def main(p_train, p_val, p_ids, chr_train, chr_val, chr_ids, out_folder, 
         batch_size, n_workers, random_seed, train_model):
    start_time = time.time()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(f'{out_folder}/logs'):
        os.makedirs(f'{out_folder}/logs')
    if not os.path.exists(f'{out_folder}/pred_labels'):
        os.makedirs(f'{out_folder}/pred_labels')
    if not os.path.exists(f'{out_folder}/models'):
        os.makedirs(f'{out_folder}/models')

    # load data
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': n_workers,
              'pin_memory': False}  # speed up data transfer from CPU to GPU
    random_gen = random.default_rng(random_seed)

    transform_input = True
    if train_model == 'SquiggleNet' or train_model == 'DeepSelectNet':
        transform_input = False

    validation_generator = CustomDataLoader(p_val, chr_val, params, random_gen, p_ids, chr_ids, transform=transform_input)

    training_generator = CustomDataLoader(p_train, chr_train, params, random_gen, transform=transform_input)
    

    print(f'Class counts for training: {training_generator.get_class_counts()}')
    print(f'Class counts for validation: {validation_generator.get_class_counts()}')

    train_pos_weight = torch.as_tensor(training_generator.get_class_counts()[1] / (training_generator.get_class_counts()[0] + 1e-5), dtype=torch.float)

    val_pos_weight = torch.as_tensor(validation_generator.get_class_counts()[1] / (validation_generator.get_class_counts()[0] + 1e-5), dtype=torch.float)

    config = {
        'learning_rate': tune.loguniform(1e-3, 1e-1),
        'batch_size' : tune.choice([32, 64, 100]),
        'train_pos_weight' : train_pos_weight,
        'val_pos_weight' : val_pos_weight,
        'num_blocks' : tune.choice([2, 3, 4]),
        'num_layers' : tune.choice([4, 5, 6, 7]),
    }

    if train_model  == 'DeepSelectNet':
        config['dropout'] = tune.choice[0.01, 0.05, 0.1, 0.2, 0.25]


    # Define a TorchTrainer without hyper-parameters for Tuner
    train_config = {
        'train_model' : train_model,
        'train_gen' : training_generator,
        'val_gen' : validation_generator,
        'out_folder' : out_folder,
        'model_config' : config,
    }

    results = _tune_models(train_config, n_workers)

    #tune.run(
    #    partial(_train_tune, epochs=10, gpus=0),
    #    config=train_config,
    #    num_samples=10
    #    )

def _train_tune(config, epochs=10, gpus=0):
    if config['train_model'] == 'SquiggleNet':
        model = SquiggleNetLightning(Bottleneck, config['model_config'])
    
    callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "mean_accuracy": "val_accuracy"
    },
    on="validation_end")
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[callback])
    trainer.fit(model, config['train_gen'], config['val_gen'])


def _tune_models(train_config, n_workers):
    # The maximum training epochs
    num_epochs = 10

    # Number of sampls from parameter space
    num_samples = 10

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        #num_workers=4, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
        num_workers=1, use_gpu=False, resources_per_worker={"CPU": n_workers},
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    ray_trainer = TorchTrainer(
        _train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": train_config},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )

    return tuner.fit()

def _train_func(train_config):
    if train_config['train_model'] == 'SquiggleNet':
        model = SquiggleNetLightning(Bottleneck, train_config['model_config'])
    elif train_config['train_model']  == 'DeepSelectNet':
        model = DeepSelectNet(DSBottleneck, train_config['model_config'])
    #elif train_model == 'bnLSTM':
    #    model = bnLSTM(input_size=32, hidden_size=512, max_length=4000, learning_rate=learning_rate, batch_size=batch_size,train_pos_weight=train_pos_weight, val_pos_weight=val_pos_weight,
    #                    num_layers=1, use_bias=True, batch_first=True, dropout=0.5)


    trainer = pl.Trainer(
        default_root_dir=train_config['out_folder'],
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_config['train_gen'], train_config['val_gen'])

if __name__ == '__main__':
    main()