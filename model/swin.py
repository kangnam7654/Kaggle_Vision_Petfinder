import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from timm import create_model

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self.lr = self.cfg.train.lr
        self._criterion = self.cfg.model.loss_function
        self.save_hyperparameters(cfg)

        # 자동 opt off
        self.automatic_optimization = self.cfg.model.auto_opt

    def __build_model(self):
        self.backbone = create_model(**self.cfg.model.backbone)
        num_features = self.backbone.num_features  # 입력 수

        # training off
        if self.cfg.train.back_freeze:  # freeze backbone
            print('freezed backbone')
            self.backbone.training = False
            for child in self.backbone.children():
                for param in child.parameters():
                    param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.BatchNorm1d(num_features*2),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(num_features * 2, num_features * 2),
            nn.BatchNorm1d(num_features * 2),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(num_features * 2, num_features * 4),
            nn.BatchNorm1d(num_features * 4),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(num_features * 4, num_features * 4),
            nn.BatchNorm1d(num_features * 4),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(num_features * 4, num_features * 2),
            nn.BatchNorm1d(num_features * 2),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(num_features * 2, num_features),
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(num_features, self.cfg.model.head.out_dim_reg)
        )

    def forward(self, x):
        out = self.backbone(x)
        # out = self.fc(out)
        return out

    def manual_opt(self, train_loss):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()
        sch.step()

    def training_step(self, batch, batch_idx):
        train_loss, train_rmse, train_pred, train_labels, train_logits = self.__share_step(batch)
        train_return = {'loss': train_loss, 'pred': train_pred, 'labels': train_labels, 'logits': train_logits}

        # manual opt
        if not self.automatic_optimization:
            self.manual_opt(train_loss)

        # log add
        self.log(f'loss', train_loss, prog_bar=True)
        self.log(f'rmse', train_rmse, prog_bar=True)
        return train_return

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_rmse, valid_pred, valid_labels, valid_logits = self.__share_step(batch)
        valid_return = {'loss': valid_loss, 'pred': valid_pred, 'labels': valid_labels, 'logits': valid_logits}
        return valid_return

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels = batch
        logits = self.forward(images)
        preds = logits.sigmoid().detach().cpu() * 100
        return preds

    def __share_step(self, batch):
        images, labels = batch
        labels = labels.float()
        logits = self.forward(images).squeeze(1)

        loss = self._criterion(logits, labels)
        pred = logits.sigmoid().detach().cpu() * 100.

        labels = labels.detach().cpu() * 100.
        rmse = self.compute_rmse(pred, labels)
        return loss, rmse, pred, labels, logits

    def training_epoch_end(self, outputs):  # regression
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'valid')

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        losses = []
        logits = []
        for out in outputs:
            pred, label, loss, logit = out['pred'], out['labels'], out['loss'], out['logits']
            preds.append(pred)
            labels.append(label)
            losses.append(loss)
            logits.append(logit)

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        logits = torch.cat(logits)

        losses = torch.tensor(losses)
        loss_ = torch.mean(losses)

        rmse = self.compute_rmse(preds, labels)

        self.log(f'{mode}_loss', loss_)
        self.log(f'{mode}_rmse', rmse)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, **self.cfg.model.scheduler.Plateau)
        monitor = 'valid_rmse'
        to_return = {'optimizer': opt, 'lr_scheduler': sch, 'monitor': monitor}
        return to_return

    @staticmethod
    def compute_rmse(pred, labels):
        rmse = torch.sqrt(((labels - pred) ** 2).mean())
        return rmse

    @staticmethod
    def compute_accuracy(out, labels):
        max_indices = torch.argmax(out, dim=1)
        acc = (max_indices == labels).to(torch.float).mean()
        return acc

    @staticmethod
    def quantile_loss(x, y):
        q1 = 0.5
        e1 = y - x
        eq1 = torch.max(q1*e1, (q1-1)*e1)
        loss = torch.mean((eq1))
        return loss

if __name__ == '__main__':
    a = timm.list_models()
