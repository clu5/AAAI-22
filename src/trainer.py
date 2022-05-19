# ==================================================================== #
#                                                                      #
#                          TRAINER CLASS                               #
#                                                                      #
# ==================================================================== #
import numpy as np
import torch
import torchvision
from tqdm import tqdm

class Trainer():
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 criterion,
                 metrics,
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
        self.calc_metrics = lambda true, pred: {k: v(true, pred) for k, v in metrics.items()}
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
        for i, batch in enumerate(tqdm(self.train_loader)):
            image, label = batch
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

            if epoch_pred is None:
                epoch_pred = prob
            else:
                epoch_pred = np.vstack([epoch_pred, prob])

            if epoch_label is None:
                epoch_label = label
            else:
                epoch_label = np.hstack([epoch_label, label])

            del image, label, pred, loss, prob

        metrics = self.calc_metrics(epoch_label, epoch_pred)
        epoch_loss /= max(N, 1)
        return {
            'metrics': metrics,
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
            for i, batch in enumerate(tqdm(self.valid_loader)):
                image, label = batch
                image = image.to(self.device)
                label = label.to(self.device)
                pred = self.model(image)
                loss = self.criterion(pred, label)
                epoch_loss += loss.detach().item()

                prob = torch.softmax(pred, 1)
                prob = prob.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                if epoch_pred is None:
                    epoch_pred = prob
                else:
                    epoch_pred = np.vstack([epoch_pred, prob])

                if epoch_label is None:
                    epoch_label = label
                else:
                    epoch_label = np.hstack([epoch_label, label])

                del image, label, pred, loss, prob

            metrics = self.calc_metrics(epoch_label, epoch_pred)
            epoch_loss /= max(N, 1)
            self.scheduler.step(epoch_loss)
        return {
            'metrics': metrics,
            'loss': epoch_loss,
            'pred': epoch_pred,
            'label': epoch_label,
        }

    def monte_carlo_inference(self, test_loader, num_iter=10):
        """ Performs monte carlo dropout inference """
        self.model.eval()
        with torch.no_grad():
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()  # Activate dropout
            mc_res = {}
            test_ds = test_loader.dataset
            for j in tqdm(range(len(test_ds))):
                image, label, meta = test_ds.__getitem__(j, return_meta_data=True)
                mc_res[meta['image']] = {}
                mc_res[meta['image']]['meta'] = meta
                for i in range(num_iter):
                    logits = self.model(image[None].to(self.device))
                    mc_res[meta['image']][f'mc_{i}'] = logits.cpu().detach().tolist()
                    #prob = torch.softmax(logits, 1)
                    #mc_res[meta['image']][f'mc_{i}'] = prob.cpu().detach().tolist()
        return mc_res

    def save(self, file_path, epoch=None):
        # for dataparallel
        if 'module' in dir(self.model): model = self.model.module
        else: model = self.model
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }, file_path)

