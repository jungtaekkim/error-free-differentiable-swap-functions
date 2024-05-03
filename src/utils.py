import numpy as np
import torch


def avg_list_of_dicts(list_of_dicts):
    result = {}
    for k in list_of_dicts[0]:
        result[k] = np.mean([d[k] for d in list_of_dicts])
    return result


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break

def ranking_accuracy(model, data, targets):
    scores = model(data).squeeze(2)

    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)

    acc_em = acc.all(-1).float().mean()
    acc_ew = acc.float().mean()

    # EM5:
    scores = scores[:, :5]
    targets = targets[:, :5]
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    acc_em5 = acc.all(-1).float().mean()

    return dict(
        acc_em=acc_em.type(torch.float32).mean().item(),
        acc_ew=acc_ew.type(torch.float32).mean().item(),
        acc_em5=acc_em5.type(torch.float32).mean().item(),
    )
