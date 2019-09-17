import torch


class CELoss(torch.nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        label = torch.zeros(target.shape[0], target.shape[1]).cuda().long()
        indices = (target == 1).nonzero()
        label[indices[:, 0], indices[:, 1]] = indices[:, -1]
        loss = 0
        for i in range(target.shape[1]):
            loss += self.cross_entropy(input[:, i, :], label[:, i])
        return loss
