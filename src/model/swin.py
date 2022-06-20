import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from timm import create_model


class SwinModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self.learning_rate = 0.01
        self._criterion = nn.MSELoss()
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(**self.cfg['model']['backbone'])
        self.head = nn.Linear(1024, 1)

        # # training off
        # if self.cfg.train.back_freeze:  # freeze backbone
        #     print('freezed backbone')
        #     self.backbone.training = False
        #     for child in self.backbone.children():
        #         for param in child.parameters():
        #             param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out

    def training_step(self, batch, batch_idx):
        train_loss = self.__share_step(batch)
        train_rmse = train_loss * 100
        results = {'loss': train_loss}
        # log add
        self.log(f'train_rmse', train_rmse)
        return results

    def validation_step(self, batch, batch_idx):
        valid_loss = self.__share_step(batch)
        valid_rmse = valid_loss * 100
        results = {'loss': valid_loss}
        # logging
        self.log(f'valid_rmse', valid_rmse)
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        logits = self.forward(images)
        preds = logits.sigmoid().detach().cpu() * 100
        return preds

    def __share_step(self, batch):
        images, labels = batch
        labels = labels.float()  # [batch_size]
        logits = self.forward(images).squeeze(1)  # [batch_size, 1] -> [batch_size]
        
        loss = self.compute_rmse(logits, labels)
        return loss

    def training_epoch_end(self, outputs):  # regression
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'valid')

    def __share_epoch_end(self, outputs, mode):
        all_loss = []
        for out in outputs:
            all_loss.append(out['loss'])
        avg_loss = torch.mean(torch.stack(all_loss))
        avg_rmse = avg_loss.item() * 100
        self.log(f'{mode}_avg_rmse', avg_rmse)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
        sch = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=opt, T_0=512, T_mult=2, eta_min=(self.learning_rate/100)),
               'interval': 'step',
               'frequency': 1}
        returns = {'optimizer': opt, 'lr_scheduler': sch}
        return returns

    def compute_rmse(self, pred, labels):
        rmse = torch.sqrt(self._criterion(pred, labels))
        # rmse = torch.sqrt(((labels - pred) ** 2).mean())
        return rmse


if __name__ == '__main__':
    print(timm.list_models())