import torch.nn as nn

class DeepMultiLabelNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(DeepMultiLabelNet, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(input_dim, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Dropout(0.4),

      nn.Linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(0.4),

      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(0.3),

      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.2),

      nn.Linear(128, output_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.net(x)