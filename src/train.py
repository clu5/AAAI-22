# ==================================================================== #
#                                                                      #
#                          TRAINER CLASS                               #
#                                                                      #
# ==================================================================== #
import numpy as np
import torch
import torchvision

class Trainer():
    def __init__(self, 
                 model,
                 optimizer,
                 scheduler,
                 criterion,
                 metric,
                 train_loader = None,
                 valid_loader = None,
                 test_loader = None,
                 device = 'cuda',
                ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metric = metric
        self.test_loader = test_loader
        self.device = device
    
    def train(self, train_loader=None):
        if train_loader is not None:
            self.train_loader = train_loader
        assert self.train_loader is not None, 'Requires train loader'
        epoch_loss, epoch_metric = 0, 0
        epoch_pred, epoch_label = None, None
        N = len(self.train_loader)
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            image, label = batch['image'], batch['label']
            image = image.to(self.device)
            label = label.to(self.device)
            pred = self.model(image)
            loss = self.criterion(pred, label)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach().item()
            
            prob = torch.softmax(pred, 1)
            prob = prob.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            
            if epoch_pred is None: epoch_pred = prob
            else: epoch_pred = np.vstack([epoch_pred, prob])
            if epoch_label is None: epoch_label = label
            else: epoch_label = np.hstack([epoch_label, label])
            
            del image, label, pred, loss, prob
            
        metric = self.metric(epoch_label, epoch_pred)
        epoch_loss /= max(N, 1)
        return {
            'metric': metric,
            'loss': epoch_loss,
        }
    
    def validate(self, valid_loader=None):
        if valid_loader is not None:
            self.valid_loader = valid_loader
        assert self.valid_loader is not None, 'Requires valid loader'
        with torch.no_grad():
            epoch_loss, epoch_metric = 0, 0
            epoch_pred, epoch_label = None, None
            N = len(self.valid_loader)
            self.model.eval()
            for i, batch in enumerate(self.valid_loader):
                image, label = batch['image'], batch['label']
                image = image.to(self.device)
                label = label.to(self.device)
                pred = self.model(image)
                loss = self.criterion(pred, label)
                epoch_loss += loss.detach().item()

                prob = torch.softmax(pred, 1)
                prob = prob.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                if epoch_pred is None: epoch_pred = prob
                else: epoch_pred = np.vstack([epoch_pred, prob])
                if epoch_label is None: epoch_label = label
                else: epoch_label = np.hstack([epoch_label, label])

                del image, label, pred, loss, prob

            metric = self.metric(epoch_label, epoch_pred)
            epoch_loss /= max(N, 1)
            self.scheduler.step(epoch_loss)
        return {
            'metric': metric,
            'loss': epoch_loss,
        }
    
    def evaluate(self, test_loader=None):
        if test_loader is not None:
            self.test_loader = test_loader
        assert self.test_loader is not None, 'Requires test loader'
        with torch.no_grad():
            epoch_loss, epoch_metric = 0, 0
            epoch_pred, epoch_label = None, None
            N = len(self.test_loader)
            self.model.eval()
            for i, batch in enumerate(self.test_loader):
                image, label = batch['image'], batch['label']
                image = image.to(self.device)
                label = label.to(self.device)
                pred = self.model(image)
                loss = self.criterion(pred, label)
                epoch_loss += loss.detach().item()

                prob = torch.softmax(pred, 1)
                prob = prob.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                if epoch_pred is None: epoch_pred = prob
                else: epoch_pred = np.vstack([epoch_pred, prob])
                if epoch_label is None: epoch_label = label
                else: epoch_label = np.hstack([epoch_label, label])

                del image, label, pred, loss, prob

            metric = self.metric(epoch_label, epoch_pred)
            epoch_loss /= max(N, 1)
        return {
            'metric': metric,
            'loss': epoch_loss,
            'pred': epoch_pred.tolist(),
            'label': epoch_label.tolist(),
        }
    
    def save(self, file_path, epoch=None):
        # for dataparallel
        if 'module' in dir(self.model): model = self.model.module
        else: model = self.model
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }, file_path)
    
