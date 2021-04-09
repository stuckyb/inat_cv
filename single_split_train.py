# Implements training on a single train/validation split.

import os
import sys
import pathlib
from model import ENModel, train_transform, val_transform
from data import getDatasets, getDataLoaders
import numpy as np
import torch
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
    '-w', '--model_wts', type=str, required=False, default='',
    help='The path to a model weights/checkpoint file.  If provided, the '
    'model will be initialized using these weights prior to training.'
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
    '-t', '--top_only', action='store_true',
    help='Train only the output layer of the model.'
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
    '-o', '--output_dir', type=str, required=True,
    help='Location for output logs and checkpoints.'
)
argp.add_argument(
    '-e', '--experiment_name', type=str, required=False, default='',
    help='Experiment name for TensorBoard (default: LABELSCOL_LR).'
)
argp.add_argument(
    '-g', '--n_gpus', type=int, required=False, default=-1,
    help='The number of GPUs to use (default: all available).'
)
argp.add_argument(
    '-p', '--no_prog_bar', action='store_true',
    help='Disables the progress bar.'
)
argp.add_argument(
    '-r', '--rand_seed', type=int, required=False, default=None,
    help='A random number seed to use.'
)

args = argp.parse_args()

if args.rand_seed is not None:
    rng = np.random.default_rng(seed=args.rand_seed)
    pl.utilities.seed.seed_everything(seed=args.rand_seed)
else:
    rng = None

outputdir = args.output_dir

if os.path.isdir(outputdir):
    exit(f'\nThe output folder {outputdir} already exists.\n')

exp_name = args.experiment_name
if exp_name == '':
    exp_name = '{0}_{1}'.format(args.labels_col, args.learning_rate)

if args.no_prog_bar:
    pb_refresh_rate = 0
else:
    pb_refresh_rate = 1

outpath = pathlib.Path(outputdir)
outpath.mkdir()

# Log the script arguments.
with open(outpath / 'training_args.txt', 'w') as fout:
    fout.write(' '.join(sys.argv) + '\n')
    fout.write('\n')
    argsdict = vars(args)
    for key in sorted(argsdict.keys()):
        fout.write('{0}: {1}\n'.format(key, argsdict[key]))

n_gpus = args.n_gpus
if n_gpus < 0:
    n_gpus = torch.cuda.device_count()

train_data, val_data = getDatasets(
    args.labels_csv, args.images, args.fnames_col, args.labels_col,
    train_size=args.train_split,
    train_transform=train_transform, val_transform=val_transform, rng=rng
)
trainloader, valloader = getDataLoaders(
    train_data, val_data, batch_size=args.batch_size,
    use_weighted_sampling=True
)

if args.model_wts != '':
    print(f'Loading model weights from {args.model_wts}...')
    model = ENModel.load_from_checkpoint(
        args.model_wts, lr=args.learning_rate,
        n_classes=len(train_data.dataset.classes)
    )
else:
    model = ENModel(args.learning_rate, len(train_data.dataset.classes))

if args.top_only:
    model.setTrainTopOnly()

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
    gpus=n_gpus,
    checkpoint_callback=checkpoint_callback,
    progress_bar_refresh_rate=pb_refresh_rate,
    max_epochs=1000,
    #accelerator='dp'
)

trainer.fit(model, trainloader, valloader)

