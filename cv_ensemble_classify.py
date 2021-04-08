
from model import ENModel, val_transform
from data import UnlabeledImagesDataset, ImageCsvDataset
import torch
import torch.utils.data
import torch.nn.functional as F
import csv
import re
from pathlib import Path
import os.path
from pytorch_lightning.metrics.classification import ConfusionMatrix
from argparse import ArgumentParser


argp = ArgumentParser(
    description='Uses a cross-validation ensemble to analyze images.'
)
argp.add_argument(
    '-c', '--cv_dir', type=str, required=True, 
    help='The path to a cross-validation output directory.'
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

# Get the cross-validation model checkpoints.
ckpts = []
cv_dir = Path(args.cv_dir)
for fold_dir in cv_dir.glob('fold_*'):
    if fold_dir.is_dir():
        best_epoch = -1
        best_ckpt = ''
        for ckpt in fold_dir.glob('*.ckpt'):
            m = re.search(r'epoch=([0-9]+)', str(ckpt))
            if m is not None:
                epoch = int(m.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_ckpt = str(ckpt)

        ckpts.append(best_ckpt)

if args.output != '':
    writer = csv.DictWriter(
        open(args.output, 'w'),
        ['file', 'prediction', '0', '1']
    )
    writer.writeheader()
else:
    writer = None

rowout = {}
models = []

with torch.no_grad():
    for i, ckpt in enumerate(ckpts):
        print(f'Loading best model from fold {i}...')
        model = ENModel.load_from_checkpoint(ckpt, lr=0.001, n_classes=2)
        model.to(device)
        model.eval()
        models.append(model)

    c_mat = ConfusionMatrix(num_classes=2)
    img_cnt = 0
    correct_cnt = 0

    for batch, labels in dl:
        # Get the predictions for the batch from each model.
        outputs = []
        for model in models:
            output = model(batch.to(device)).cpu()
            output = F.softmax(output, 1)
            #print(output)
            outputs.append(output)

        # For each image in the batch, average the model predictions.
        p_labels = torch.zeros_like(labels)
        for i in range(len(labels)):
            # Gather the predictions for this image.
            img_preds = []
            for j in range(len(models)):
                img_preds.append(outputs[j][i,:])

            # Calculate the ensemble prediction for this image.
            img_preds = torch.stack(img_preds)
            #print(img_preds)
            model_avg = torch.mean(img_preds, 0)
            #print(model_avg)
            p_label = torch.max(model_avg, 0).indices
            p_labels[i] = p_label
            #print(label, int(p_label), imgfile)
            #print(p_labels, all_images.samples[i])

        #print(p_labels)

        # Make adjustments to the ground truth labels for unclassifiable and
        # mixed images.
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

        if writer is not None:
            rowout['file'] = os.path.basename(imgfile)
            rowout['prediction'] = int(p_label)
            rowout['0'] = float(model_avg[0])
            rowout['1'] = float(model_avg[1])
            writer.writerow(rowout)

        c_mat.update(p_labels, adj_labels)
        img_cnt += len(adj_labels)
        correct_cnt += sum(p_labels == adj_labels)
        print(correct_cnt / img_cnt, img_cnt)

print('\nTotal images:', img_cnt)
print('Accuracy:', float(correct_cnt / img_cnt))
print('Confusion matrix:')
print(c_mat.compute().numpy())

