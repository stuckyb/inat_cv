
from model import ENModel, val_transform
from data import UnlabeledImagesDataset, ImageCsvDataset
import sys
import torch
import torch.utils.data
import torch.nn.functional as F
import os.path
import pandas as pd
import csv 
import numpy as np
from pytorch_lightning.metrics.classification import ConfusionMatrix


from argparse import ArgumentParser


argp = ArgumentParser(
    description='Uses a single model to analyze images.'
)
argp.add_argument(
    '-m', '--model_wts', type=str, required=True,
    help='The path to a model weights/checkpoint file.'
)
argp.add_argument(
    '-i', '--images', type=str, required=True,
    help='The path to a collection of images.'
)
argp.add_argument(
    '-l', '--labels_csv', type=str, required=False, default='',
    help='The path of a labels CSV file. If not provided, no accuracy or '
    'other metrics will be calculated.'
)
argp.add_argument(
    '-x', '--fnames_col', type=str, required=False, default='',
    help='The column in the CSV file containing the image file names.'
)
argp.add_argument(
    '-y', '--labels_col', type=str, required=False, default='',
    help='The column in the CSV file containing the image labels.'
)
argp.add_argument(
    '-b', '--batch_size', type=int, required=False, default=8,
    help='The batch size.'
)
argp.add_argument(
    '-o', '--output', type=str, required=False, default='',
    help='The path of an output CSV file.'
)

args = argp.parse_args()

if args.labels_csv != '' and (args.fnames_col == '' or args.labels_col == ''):
    exit(
        '\nError: If a labels CSV file is provided, file and label column '
        'names must also be provided.\n'
    )

if args.labels_csv == '':
    imgs_ds = UnlabeledImagesDataset(args.images, transform=val_transform)
else:
    imgs_ds = ImageCsvDataset(
        args.labels_csv, args.images, args.fnames_col, args.labels_col,
        transform=val_transform
    )

dl = torch.utils.data.DataLoader(
    imgs_ds, batch_size=args.batch_size, shuffle=False, num_workers=12
)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if args.output != '':
    writer = csv.DictWriter(
        open(args.output, 'w'),
        ['file', 'prediction', '0', '1']
    )
    writer.writeheader()
else:
    writer = None

rowout = {}

with torch.no_grad():
    model = ENModel.load_from_checkpoint(args.model_wts, lr=0.001, n_classes=2)
    model.to(device)
    model.eval()
    #print(model.feature_extractor._fc.weight)

    c_mat = ConfusionMatrix(num_classes=2)
    img_cnt = 0
    correct_cnt = 0

    for batch, labels in dl:
        outputs = model(batch.to(device)).cpu()
        p_labels = torch.max(outputs, 1).indices

        # Make adjustments for unclassifiable and mixed images.
        adj_labels = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            if imgs_ds.classes[label] == 'U':
                # An unclassifiable image; the adjustment should be the
                # opposite of the prediction (i.e., it is always wrong).
                adj_labels[i] = abs(p_labels[i] - 1)
            elif imgs_ds.classes[label] == 'B':
                # An image containing both color morphs.
                adj_labels[i] = p_labels[i]
            else:
                adj_labels[i] = labels[i]
                
        #print(labels, adj_labels)
        #print(p_labels, adj_labels)
        #print(p_labels == adj_labels)
        #print(sum(p_labels == adj_labels))

        if writer is not None:
            sm_outputs = F.softmax(outputs, 1)
            for i, p_label in enumerate(p_labels):
                rowout['file'] = os.path.basename(imgs_ds.x[img_cnt + i])
                rowout['prediction'] = int(p_label)
                rowout['0'] = float(sm_outputs[i][0])
                rowout['1'] = float(sm_outputs[i][1])
                writer.writerow(rowout)

        c_mat.update(p_labels, adj_labels)
        img_cnt += len(adj_labels)
        correct_cnt += sum(p_labels == adj_labels)
        print(correct_cnt / img_cnt, img_cnt)

print('\nTotal images:', img_cnt)
print('Accuracy:', float(correct_cnt / img_cnt))
print('Confusion matrix:')
print(c_mat.compute().numpy())

