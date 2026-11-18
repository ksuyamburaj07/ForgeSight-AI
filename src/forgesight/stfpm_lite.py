"""Compact student network and normalized feature-matching loss."""

import torch
from torch import nn
from torch.nn import functional as F


class CompactStudent(nn.Module):
    """A lightweight CNN with ResNet-18 layer2/layer3 output geometry."""

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fine = nn.Conv2d(64, 128, 1)

        self.coarse = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError("Expected N x 3 x 224 x 224 input.")

        hidden = self.stem(x)

        return (
            self.fine(hidden),
            self.coarse(hidden),
        )


def normalized_feature_loss(student_features, teacher_features):
    """Mean squared normalized-feature discrepancy at two scales."""

    if len(student_features) != 2 or len(teacher_features) != 2:
        raise ValueError("Expected exactly two feature stages.")

    losses = []

    for student, teacher in zip(student_features, teacher_features):
        if student.ndim != 4 or student.shape != teacher.shape:
            raise ValueError("Teacher/student feature shapes differ.")

        target = F.normalize(
            teacher.detach(),
            p=2,
            dim=1,
            eps=1e-6,
        )

        predicted = F.normalize(
            student,
            p=2,
            dim=1,
            eps=1e-6,
        )

        losses.append(
            (predicted - target).square().sum(dim=1).mean()
        )

    return 0.5 * (losses[0] + losses[1])
