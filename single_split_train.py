# Implements training on a single train/validation split.

import os
from model import ENModel, train_transform, val_transform
from data import getDatasets, getDataLoaders
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Trains on a single train/validation split.'
)
argp.add_argument(
    '-i', '--images', type=str, required=True,
    help='The path to a folder of images.'
)
argp.add_argument(
    '-c', '--labels_csv', type=str, required=True,
    help='The path of a CSV file with image labels.'
)
argp.add_argument(
    '-x', '--fnames_col', type=str, required=True,
    help='The column in the CSV file containing the image file names.'
)
argp.add_argument(
    '-y', '--labels_col', type=str, required=True,
    help='The column in the CSV file containing the image labels.'
)
argp.add_argument(
    '-l', '--learning_rate', type=float, required=False, default=0.001,
    help='The learning rate.'
)
argp.add_argument(
    '-s', '--train_split', type=float, required=False, default=0.75,
    help='The proportion of images to use for training.'
)
argp.add_argument(
    '-b', '--batch_size', type=int, required=False, default=8,
    help='The batch size.'
)
argp.add_argument(
    '-m', '--max_iters', type=int, required=False, default=20000,
    help='The maximum number of training iterations (default: 20,000).'
)
argp.add_argument(
    '-o', '--output_dir', type=str, required=False, default='',
    help='Location for output logs and checkpoints (default: cwd).'
)
argp.add_argument(
    '-e', '--experiment_name', type=str, required=False, default='',
    help='Experiment name for TensorBoard (default: LABELSCOL_LR).'
)
argp.add_argument(
    '-r', '--rand_seed', type=int, required=False, default=None,
    help='A random number seed to use.'
)

args = argp.parse_args()

if args.rand_seed is not None:
    rng = np.random.default_rng(seed=args.rand_seed)
else:
    rng = None

outputdir = args.output_dir
if outputdir == '':
    outputdir = os.getcwd()

exp_name = args.experiment_name
if exp_name == '':
    exp_name = '{0}_{1}'.format(args.labels_col, args.learning_rate)


train_data, val_data = getDatasets(
    args.labels_csv, args.images, args.fnames_col, args.labels_col,
    train_size=args.train_split,
    train_transform=train_transform, val_transform=val_transform, rng=rng
)
trainloader, valloader = getDataLoaders(
    train_data, val_data, batch_size=args.batch_size
)

model = ENModel(args.learning_rate, len(train_data.dataset.classes))
tb_logger = pl_loggers.TensorBoardLogger(outputdir, exp_name)

checkpoint_callback = ModelCheckpoint(
    dirpath=outputdir,
    save_top_k=1,
    verbose=True,
    monitor='valid_loss',
    mode='min',
    prefix='weights'
)

trainer = pl.Trainer(
    logger=tb_logger, max_steps=args.max_iters,
    gpus=1, checkpoint_callback=checkpoint_callback
)
trainer.fit(model, trainloader, valloader)

