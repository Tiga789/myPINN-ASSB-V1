from dataclasses import dataclass

import torch


@dataclass
class Struct:
    dummy: int = 0


def lbfgs(
    loss_and_flat_grad,
    weights,
    state,
    model=None,
    bestLoss=None,
    modelFolder=None,
    maxIter=1,
    learningRate=1.0,
    dynamicAttention=False,
    logLossFolder=None,
    nEpochDoneLBFGS=0,
    nEpochDoneSGD=0,
    nBatchSGD=1,
    nEpochs_start_lbfgs=0,
):
    if model is None:
        raise ValueError("model must be provided")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=float(learningRate),
        max_iter=int(maxIter),
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_and_flat_grad()
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss, dtype=torch.float64)
        loss.backward()
        loss_history.append(float(loss.detach().cpu()))
        return loss

    optimizer.step(closure)
    current_best = min(loss_history) if loss_history else None
    if bestLoss is None:
        bestLoss = current_best
    elif current_best is not None:
        bestLoss = min(float(bestLoss), float(current_best))
    return model, loss_history, state, bestLoss
