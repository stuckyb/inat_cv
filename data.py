
import torch
import numpy as np
import csv
import os.path
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image


class ImageCsvDataset(Dataset):
    """
    An images Dataset with image locations and classifications obtained from a
    CSV file.
    """
    def __init__(self, csv_file, root_dir, x, y, transform=None):
        """
        csv_file (str): Path of a CSV file with image file names and labels.
        root_dir (str): Folder containing the image files.
        x (str): Column containing the image file names.
        y (str): Column containing the image labels.
        transform: Optional transform to be applied to each image.
        """
        self.xcol = x
        self.ycol = y
        self.classes = None
        self.x = None
        self.y = None
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

        self._loadData(self.csv_file, self.xcol, self.ycol)

    def _loadData(self, csv_file, xcol, ycol):
        self.x = []
        labels = []
        labelset = set()

        with open(csv_file) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                labels.append(row[ycol])
                labelset.add(row[ycol])
                self.x.append(row[xcol])

        self.classes = list(sorted(labelset))

        # Convert the str labels to integer class indices.
        class_str_to_i = {}
        for i, class_str in enumerate(self.classes):
            class_str_to_i[class_str] = i

        self.y = []
        for label_str in labels:
            self.y.append(class_str_to_i[label_str])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.x[idx])
        img = Image.open(img_name)

        if self.transform is not None:
            img = self.transform(img)

        return (img, self.y[idx])


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    labels = []
    for cnt, item in enumerate(images):
        count[item[1]] += 1
        labels.append(item[1])
        print(cnt)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight

# Transformations for training data.
train_transform = transforms.Compose([
        transforms.Resize((596,447)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomAffine(degrees=0, translate=(.1, .1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Transformations for validation/test data.
val_transform = transforms.Compose([
        transforms.Resize((596,447)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def getAllImagesDataset(path):
    """
    Returns a DataSet instance with all images in a target folder and constant
    "dummy" label for each image.
    """
    all_images = torchvision.datasets.ImageFolder(
        root=path, transform=val_transform
    )

    return all_images


def getDatasets(path, train_size=0.75, rng=None, train_idx=None, valid_idx=None):
    if rng is None:
        rng = np.random.default_rng()
   
    trainset = torchvision.datasets.ImageFolder(
        root=path, transform=train_transform
    )
    validset = torchvision.datasets.ImageFolder(
        root=path, transform=val_transform
    )   
   
    if train_idx is None or valid_idx is None:
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(train_size * num_train))
        rng.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        
        traindata = torch.utils.data.Subset(trainset, indices=train_idx)
        valdata = torch.utils.data.Subset(validset, indices=valid_idx)
    else:
        traindata = torch.utils.data.Subset(trainset, indices=train_idx)
        valdata = torch.utils.data.Subset(validset, indices=valid_idx)
    
    return traindata, valdata


def getDataLoaders(
    train_data, val_data, batch_size=8, num_workers=16,
    use_weighted_sampling=True
):
    if use_weighted_sampling:
        # For unbalanced dataset we create a weighted sampler
        weights = make_weights_for_balanced_classes(
            train_data, len(train_data.dataset.classes))
        weights = torch.Tensor(weights)
        sampler1 = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights)
        )
        trainloader = torch.utils.data.DataLoader(
            train_data, sampler=sampler1, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )

    valloader = torch.utils.data.DataLoader(
        val_data, batch_size=8, num_workers=16
    )

    return trainloader, valloader

