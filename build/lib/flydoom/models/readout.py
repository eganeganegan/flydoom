"""Small readouts restricted to selected descending neurons."""

from torch import nn


class DescendingReadout(nn.Module):
    def __init__(self, descending_count: int, output_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(descending_count, output_count)

    def forward(self, descending_activity):
        return self.linear(descending_activity)
