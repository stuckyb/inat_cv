import os
from model import ENModel, train_transform, val_transform
from data import getDatasets, getDataLoaders, ImageCsvDataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import os.path
import pathlib
from sklearn.model_selection import StratifiedKFold
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Trains using stratified k-fold cross-validation.'
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
    '-k', '--n_folds', type=int, required=False, default=4,
    help='The number of cross-validation folds.'
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
else:
    rng = None

outputdir = args.output_dir

exp_name = args.experiment_name
if exp_name == '':
    exp_name = '{0}_{1}'.format(args.labels_col, args.learning_rate)


if os.path.exists(outputdir):
    exit(f'\nThe output folder {outputdir} already exists.\n')

if args.no_prog_bar:
    pb_refresh_rate = 0
else:
    pb_refresh_rate = 1

outpath = pathlib.Path(outputdir)
outpath.mkdir()

# Log the script arguments.
with open(outpath / 'training_args.txt', 'w') as fout:
    argsdict = vars(args)
    for key in sorted(argsdict.keys()):
        fout.write('{0}: {1}\n'.format(key, argsdict[key]))

kf = StratifiedKFold(
    n_splits=args.n_folds, shuffle=True, random_state=args.rand_seed
)
all_images = ImageCsvDataset(
    args.labels_csv, args.images, args.fnames_col, args.labels_col
)

loop_count = 0
for train_idx, valid_idx in kf.split(all_images.x, all_images.y): 
    #print('train_idx: {0}, valid_idx: {1}'.format(train_idx, valid_idx))
    #print(len(train_idx), len(valid_idx))

    fold_folder = outpath / ('fold_' + str(loop_count))
    fold_folder.mkdir()
    print(
        f'\n#\n# Cross-validation fold {loop_count}; saving results to '
        f'{fold_folder}.\n#'
    )
    
    train_data, val_data = getDatasets(
        args.labels_csv, args.images, args.fnames_col, args.labels_col,
        train_idx=train_idx, valid_idx=valid_idx,
        train_transform=train_transform, val_transform=val_transform, rng=rng
    )
    trainloader, valloader = getDataLoaders(
        train_data, val_data, batch_size=args.batch_size
    )

    if args.model_wts != '':
        print(f'Loading model weights from {args.model_wts}...')
        model = ENModel.load_from_checkpoint(
            args.model_wts, lr=args.learning_rate,
            n_classes=len(train_data.dataset.classes)
        )
    else:
        model = ENModel(args.learning_rate, len(train_data.dataset.classes))

    tb_logger = pl_loggers.TensorBoardLogger(
        str(outpath), exp_name + '-fold_{0}'.format(loop_count)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(fold_folder),
        save_top_k=1,
        verbose=True,
        monitor='valid_loss',
        mode='min',
        prefix=f'weights'
    )

    trainer = pl.Trainer(
        logger=tb_logger, max_steps=args.max_iters,
        gpus=1, checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=pb_refresh_rate,
        max_epochs=10000
    )

    trainer.fit(model, trainloader, valloader)

    loop_count += 1

