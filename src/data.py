import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import math

def spatial_and_temporal_ym(patterns):
    ts = torch.arange(0.0, 32, 1.0)
    cols = [torch.Tensor() for _ in range(16)]
    for p in patterns:
        tp, sp = p
        a = -1.0
        for i in range(16):
            if i % sp == 0:
                a *= -1.0
            data = a * torch.cos(2.0 * math.pi * tp * ts)
            cols[i] = torch.cat([cols[i], data])
    return cols

def prepare_yamaguchi(args):
    spatial_patterns = [2, 4, 8]
    temporal_patterns = [1.0/s for s in spatial_patterns]
    all_pattern = []
    for sp in spatial_patterns:
        for tp in temporal_patterns:
            all_pattern.append((tp, sp))
    waves = spatial_and_temporal_ym(all_pattern)
    waves = torch.stack(waves).T.view(-1, 32, 16)

    train_x = [waves]
    train_y = [(torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long))]

    train_x = [x.to(args.device) for x in train_x]
    train_y = [(sp_y.to(args.device), tp_y.to(args.device)) for sp_y, tp_y in train_y]

    train_loader = [(x, y) for x, y in zip(train_x, train_y)]

    test_x = waves.clone().to(args.device)
    test_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long).to(args.device), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long).to(args.device)

    return train_loader, test_x, test_y