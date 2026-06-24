"""Unit tests for the CORAL ordinal head — run with: pytest tests/"""
import torch
import pytest

from src.models.ordinal_head import (
    CoralHead, coral_targets, coral_loss,
    coral_logits_to_grade, coral_grade_confidence,
)


def test_coral_targets_structure():
    grades  = torch.tensor([0, 2, 4])
    targets = coral_targets(grades, num_grades=5)
    expected = torch.tensor([
        [0., 0., 0., 0.],   # grade 0
        [1., 1., 0., 0.],   # grade 2
        [1., 1., 1., 1.],   # grade 4
    ])
    assert torch.equal(targets, expected)


def test_rank_consistency():
    """CORAL's shared-weight design must guarantee monotonic threshold probs."""
    head = CoralHead(in_features=16, num_grades=5)
    x = torch.randn(64, 16)
    probas = torch.sigmoid(head(x))
    # P(grade > k) must be non-increasing in k for every sample
    diffs = probas[:, 1:] - probas[:, :-1]
    assert (diffs <= 1e-6).all(), "Rank consistency violated"


def test_grade_decoding_roundtrip():
    # Build logits that decisively encode grade 3: yes,yes,yes,no
    logits = torch.tensor([[5.0, 5.0, 5.0, -5.0]])
    assert coral_logits_to_grade(logits).item() == 3


def test_loss_decreases_with_training():
    torch.manual_seed(0)
    head = CoralHead(in_features=8, num_grades=5)
    opt  = torch.optim.Adam(head.parameters(), lr=0.05)
    x = torch.randn(256, 8)
    # LEARNABLE labels: grade is a function of the first feature, so a linear
    # head can actually fit it. (Random labels have an irreducible loss floor.)
    score = x[:, 0]
    y = torch.bucketize(score, torch.tensor([-1.0, -0.3, 0.3, 1.0])).clamp(0, 4)
    first = None
    for step in range(100):
        loss = coral_loss(head(x), y)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.6, (
        f"CORAL loss failed to fit learnable data: {first:.3f} → {loss.item():.3f}"
    )


def test_confidence_range():
    logits = torch.randn(32, 4) * 3
    conf = coral_grade_confidence(logits)
    assert (conf >= 0).all() and (conf <= 1).all()


def test_biases_initialise_like_linspace():
    head = CoralHead(in_features=8, num_grades=5)
    assert torch.allclose(head.biases, torch.linspace(2.0, -2.0, 4), atol=1e-5)


def test_rank_consistency_survives_training():
    """Biases must stay strictly descending AFTER optimizer updates, not just
    at initialisation — this is the structural guarantee of the softplus
    reparameterization."""
    torch.manual_seed(1)
    head = CoralHead(in_features=8, num_grades=5)
    opt  = torch.optim.Adam(head.parameters(), lr=0.5)  # aggressive on purpose
    for _ in range(50):
        x = torch.randn(64, 8)
        y = torch.randint(0, 5, (64,))
        loss = coral_loss(head(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

    biases = head.biases.detach()
    assert (biases[:-1] > biases[1:]).all(), f"Biases not descending: {biases}"

    probas = torch.sigmoid(head(torch.randn(64, 8)))
    diffs = probas[:, 1:] - probas[:, :-1]
    assert (diffs <= 1e-6).all(), "Rank consistency violated after training"
