from timm import create_model
import pytorch_lightning as pl

import torch
import torch.nn as nn

class ClassSwin(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion_class = eval(self.cfg.model.loss.classification)()
        self._criterion_reg = eval(self.cfg.model.loss.reg)()
        if self.cfg.train.svr:
            self._criterion_reg = nn.HingeEmbeddingLoss()
        self.compute_acc = self.compute_accuracy
        self.save_hyperparameters(cfg)
        # 자동 opt off
        self.automatic_optimization = self.cfg.model.auto_opt

    def __build_model(self):
        self.backbone = create_model(**self.cfg.model.backbone)

        if self.cfg.train.svr: # SVR
            self.backbone.training = False
            for child in self.backbone.children():
                for param in child.parameters():
                    param.requires_grad = False
        num_features = self.backbone.num_features  # 입력 수

        if not self.cfg.train.svr:
            if self.cfg.train.reg:
                self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.head.out_dim_reg))
            else:
                self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.head.out_dim_class))

    @staticmethod
    def compute_rmse(pred, labels):
        rmse = torch.sqrt(((labels - pred) ** 2).mean())
        return rmse

    @staticmethod
    def compute_accuracy(out, labels): # for classification
        max_indices = torch.argmax(out, dim=1)
        acc = (max_indices == labels).to(torch.float).mean()
        return acc

    @staticmethod
    def compute_class(out):
        out_softmax = torch.softmax(out, dim=1)
        pred = out_softmax.argmax(dim=1).detach().cpu()
        pred = pred * 5 + 2.5
        return pred

    def forward(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        return out

    def manual_opt(self, train_loss):
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()
        sch = self.lr_schedulers()
        sch.step()

    def training_step(self, batch, batch_idx):
        if self.cfg.train.reg: # regression
            train_loss, train_rmse, train_pred, train_labels = self.__share_step_reg(batch, 'train')
            train_return = {'loss': train_loss, 'pred': train_pred, 'labels': train_labels}
        else: # classification
            train_loss, train_acc, train_rmse, train_pred, train_labels = self.__share_step_class(batch, 'train')
            train_return = {'loss': train_loss, 'acc': train_acc, 'pred': train_pred, 'labels': train_labels}
            self.log(f'acc', train_acc, prog_bar=True)

        # manual opt
        if not self.automatic_optimization:
            self.manual_opt(train_loss)

        # log add
        self.log(f'loss', train_loss, prog_bar=True)
        self.log(f'rmse', train_rmse, prog_bar=True)
        return train_return

    def validation_step(self, batch, batch_idx):
        if self.cfg.train.reg:  # regression
            valid_loss, valid_rmse, valid_pred, valid_labels = self.__share_step_reg(batch, 'valid')
            valid_return = {'loss': valid_loss, 'pred': valid_pred, 'labels': valid_labels}
        else:  # classification
            valid_loss, valid_acc, valid_rmse, valid_pred, valid_labels = self.__share_step_class(batch, 'valid')
            valid_return = {'loss': valid_loss, 'pred': valid_pred, 'labels': valid_labels, 'acc': valid_acc}
        return valid_return

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        try:
            images, labels = batch
        except:
            images = batch
        logits = self.forward(images)

        if self.cfg.train.reg:  # regression
            pred = logits.sigmoid().detach().cpu() * 100.
        else:  # classification
            pred = logits.argmax(dim=1).detach().cpu()
            pred = pred * 5 + 2
        return pred

    def __share_step_reg(self, batch, mode):
        images, labels = batch
        labels = labels.float()
        logits = self.forward(images).squeeze(1)

        loss = self._criterion_reg(logits, labels)
        pred = logits.sigmoid().detach().cpu() * 100.

        labels = labels.detach().cpu() * 100.
        rmse = self.compute_rmse(pred, labels)
        return loss, rmse, pred, labels

    def __share_step_class(self, batch, mode):
        images, labels = batch
        out = self.forward(images)

        loss = self._criterion_class(out, labels)
        acc = self.compute_acc(out, labels)
        pred = self.compute_class(out)

        labels = labels.detach().cpu()
        rmse = self.compute_rmse(pred, labels)
        return loss, acc, rmse, pred, labels

    def training_epoch_end(self, outputs):
        if self.cfg.train.reg:  # regression
            self.__share_epoch_end_reg(outputs, 'train')
        else: # classification
            self.__share_epoch_end_class(outputs, 'train')

    def validation_epoch_end(self, outputs):
        if self.cfg.train.reg:  # regression
            self.__share_epoch_end_reg(outputs, 'valid')
        else: # classification
            self.__share_epoch_end_class(outputs, 'valid')

    def __share_epoch_end_reg(self, outputs, mode):
        preds = []
        labels = []
        losses = []
        for out in outputs:
            pred, label, loss = out['pred'], out['labels'], out['loss']
            preds.append(pred)
            labels.append(label)
            losses.append(loss)

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        losses = torch.tensor(losses)
        loss_ = torch.mean(losses)

        rmse = self.compute_rmse(preds, labels)

        self.log(f'{mode}_loss', loss_)
        self.log(f'{mode}_rmse', rmse)

    def __share_epoch_end_class(self, outputs, mode):
        preds = []
        labels = []
        losses = []
        accs = []

        for out in outputs:
            pred, label, loss, acc = out['pred'], out['labels'], out['loss'], out['acc']
            preds.append(pred)
            labels.append(label)
            losses.append(loss)
            accs.append(acc)

        preds = torch.cat(preds).float()
        labels = torch.cat(labels)

        # list -> tensor
        losses = torch.tensor(losses)
        accs = torch.tensor(accs)

        # loss & accuracy
        loss_ = torch.mean(losses)
        acc_ = torch.mean(accs)

        rmse = self.compute_rmse(preds, labels)

        self.log(f'{mode}_loss', loss_)
        self.log(f'{mode}_rmse', rmse)
        self.log(f'{mode}_accuracy', acc_)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), **self.cfg.model.optimizer)
        sch = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, **self.cfg.model.scheduler.OneCycleLR)
        return [opt], [sch]

if __name__ == '__main__':
    pass
