import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet 


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


class ENModel(pl.LightningModule):
    """
    A model that uses a pretrained EfficientNet B4 as its base model.
    """
    def __init__(self, lr, n_classes):
        super().__init__()

        self.base_model = EfficientNet.from_pretrained(
            'efficientnet-b4', num_classes=n_classes
        )
        self.lr = lr
        self.n_classes = n_classes
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_conf = pl.metrics.ConfusionMatrix(num_classes=n_classes)

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        #print(outputs, labels)
        p_labels = torch.max(outputs, 1).indices
        print(p_labels, labels)
        loss = F.cross_entropy(outputs, labels)
        accuracy = self.train_acc(outputs, labels)
        print(accuracy)
        self.log_dict(
            {'train_loss': loss, 'train_acc': accuracy},
            on_step=True, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        #print(outputs, labels)
        p_labels = torch.max(outputs, 1).indices
        print("valid_labels:", p_labels, labels)
        loss = F.cross_entropy(outputs, labels)
        accuracy = self.valid_acc(outputs, labels)
        print(accuracy)
        self.valid_conf.update(p_labels, labels)
        self.log_dict(
            {'valid_loss': loss, 'valid_acc': accuracy},
            on_step=False, on_epoch=True, sync_dist=True
        )

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        print(self.valid_conf.compute())

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9) 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=8, verbose=True, factor=0.1
        )

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'valid_loss'
       }

