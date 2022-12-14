import logging
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from backbone.resnet import ResNet, ResSENet
from backbone.minimscnn import MiniMSCNN
from backbone.mobilenet import MobileNetV2


class ProtCNN(torch.nn.Module):
    def __init__(self, num_classes=17930, seq_max_len=120, backbone="resnet"):
        super().__init__()

        model = {
            "resnet": ResNet(num_classes=num_classes, ratio=seq_max_len / 120),
            "resSEnet": ResSENet(num_classes=num_classes, ratio=seq_max_len / 120),
            "MiniMSCNN": MiniMSCNN(num_classes=num_classes),
            "MobileNetV2": MobileNetV2(
                input_channel=seq_max_len,
                num_classes=num_classes,
            ),
        }
        assert backbone in model.keys()
        print("Using Model:", backbone)
        self.model = model[backbone]

    def forward(self, x):
        return self.model(x.float())


# # lightning version
# class ProtCNN_lightning(pl.LightningModule):

#     def __init__(self, num_classes, backbone = "resnet", lr = 1e-4 , weight_decay = 0.01, optimizer = "adam", ):
#         super().__init__()

#         model = {"resnet": ResNet(),
#                  "resSEnet": ResSENet()
#                 }
#         # assert backbone in model.keys()
#         print("Using Model:", backbone)
#         self.model = model[backbone]

#         self.train_acc = torchmetrics.Accuracy()
#         self.valid_acc = torchmetrics.Accuracy()
#         self.test_acc = torchmetrics.Accuracy()

#         self.lr = lr
#         self.optimizer = optimizer
#         self.weight_decay = weight_decay

#     def forward(self, x):
#         return self.model(x.float())

#     def training_step(self, batch, batch_idx):
#         x, y = batch['sequence'], batch['target']
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('train_loss', loss, on_step=True, on_epoch=True)

#         pred = torch.argmax(y_hat, dim=1)
#         acc = self.train_acc(pred, y)
#         self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
#         logging.info(f"train_acc:{acc:.4f}")
#         logging.info(f"train_loss:{loss:.4f}")

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch['sequence'], batch['target']
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('valid_loss', loss, on_step=False, on_epoch=True)

#         pred = torch.argmax(y_hat, dim=1)
#         acc = self.valid_acc(pred, y)
#         self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

#         logging.info(f"valid_acc:{acc:.4f}")
#         logging.info(f"valid_loss:{loss:.4f}")

#         return acc

#     def test_step(self, batch, batch_idx):
#         x, y = batch['sequence'], batch['target']
#         y_hat = self(x)
#         pred = torch.argmax(y_hat, dim=1)
#         acc = self.test_acc(pred, y)
#         self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

#         logging.info(f"test_acc:{acc:.4f}")

#         return acc

#     def configure_optimizers(self):
#         if self.optimizer == "adam":
#             optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
#             return {
#                 "optimizer": optimizer
#             }
#         elif self.optimizer == "sgd":
#             optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
#             return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 5], gamma=0.1),
#                 }
