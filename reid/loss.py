import torch
from torch import nn

class HardBatchTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, mean_reduction: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.mean_reduction = mean_reduction
        
    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        dist_mat = torch.cdist(features, features, p=2)
        for pid in torch.unique(targets):
            mask = targets == pid
            
            pid_dists = dist_mat[mask]
            same_pid_dists = pid_dists[:, mask]
            different_pid_dists = pid_dists[:, ~mask]
            maxs, _ = torch.max(same_pid_dists, dim=1)
            mins, _ = torch.min(different_pid_dists, dim=1)
            diff = maxs - mins
            slack = torch.clamp(diff + self.margin, min=0.0)
            
            loss += slack.sum()
            
        if self.mean_reduction:
            loss /= targets.size(0)
        
        return loss
    
losses = {
    "HARD_BATCH_TRIPLET": HardBatchTripletLoss,
    "CROSS_ENTROPY": nn.CrossEntropyLoss,
}