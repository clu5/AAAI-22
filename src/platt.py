""" Platt scaling / temperature scaling for calibration on logits """
import torch
import numpy as np


def get_platt_scaling(
        labels: np.ndarray,
        logits: np.ndarray,
        init_weight: float = 1.3,
        learning_rate: float = 5e-3,
        epsilon: float = 1e-3,
        epochs: int = 100,
        ) -> float:
    param = torch.nn.Parameter(torch.Tensor([init_weight]))
    optimizer = torch.optim.SGD([param], lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    platt_loader = torch.utils.data.TensorDataset(
        torch.Tensor(logits)[None],
        torch.Tensor(labels)[None].long()
    )

    for _ in range(epochs):
        param_old = param.item()
        for logit, label in platt_loader:
            optimizer.zero_grad()
            logit.requires_grad = True
            out = logit / param
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

        if abs(param_old - param.item()) < epsilon:
            break

    return param.item()
