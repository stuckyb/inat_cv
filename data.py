
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

    def getBalancedSampleWeights(self, indices=None):
        """
        Returns a list containing weights for each image that can be used to
        create a balanced sampler for this dataset.  If indices is provided,
        only the indices in the list are used to generate the weights.
        """
        if indices is None:
            indices = list(range(len(self)))

        n_classes = len(self.classes)

        # Get the counts of each label class.
        counts = [0] * n_classes
        for i in indices:
            counts[self.y[i]] += 1

        # Calculate the weight to assign to each class.
        weight_per_class = [0.] * n_classes
        for i in range(n_classes):
            if counts[i] != 0.:
                weight_per_class[i] = len(indices) / float(counts[i])

        # Build the list of weights for each image.
        weights = []
        for i in indices:
            weights.append(weight_per_class[self.y[i]])

        return weights


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


def getDatasets(
    csv_file, root_dir, x, y, train_size=0.75,
    train_transform=None, val_transform=None, rng=None,
    train_idx=None, valid_idx=None
):
    """
    Creates training and validation datasets for the given images folder and
    CSV file containing image labels.

    csv_file (str): Path of a CSV file with image file names and labels.
    root_dir (str): Folder containing the image files.
    x (str): Column containing the image file names.
    y (str): Column containing the image labels.
    train_transform: Optional transform to be applied to each training image.
    val_transform: Optional transform to be applied to each validation image.
    """
    if rng is None:
        rng = np.random.default_rng()
   
    trainset = ImageCsvDataset(
        csv_file, root_dir, x, y, transform=train_transform
    )
    validset = ImageCsvDataset(
        csv_file, root_dir, x, y, transform=val_transform
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
    """
    train_data: A Subset instance.
    val_data: A Subset instance.
    """
    if use_weighted_sampling:
        # Create a weighted sampler.
        base_ds = train_data.dataset
        weights = base_ds.getBalancedSampleWeights(train_data.indices)
        weights = torch.Tensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights)
        )
        trainloader = torch.utils.data.DataLoader(
            train_data, sampler=sampler, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )

    valloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers
    )

    return trainloader, valloader

