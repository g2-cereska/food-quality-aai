"""Tests for src/model_utils.py.

Uses weights=None-equivalent behaviour is not directly exposed by
build_model (it always loads ImageNet weights), so these tests focus on
what can be verified without a network call: output shape, head
replacement, parameter counting, and error handling for unknown
architectures. Downloading pretrained weights is deliberately exercised
here too (small models, cached after first run) since getting the head
replacement wrong is exactly the kind of bug that only shows up by
actually building the model.
"""
from __future__ import annotations

import pytest
import torch

from src.model_utils import SUPPORTED_ARCHITECTURES, build_model, count_parameters, get_device


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_build_model_output_shape(architecture):
    """The replaced classification head must produce exactly num_classes
    logits, for every supported architecture."""
    num_classes = 28
    model = build_model(architecture, num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, num_classes)


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_build_model_different_num_classes(architecture):
    """The head must resize correctly for a different class count, not
    just the 28 used in this project — catches accidentally hardcoding
    28 somewhere instead of using the num_classes argument."""
    model = build_model(architecture, num_classes=5)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 5)


def test_build_model_unknown_architecture_raises():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("not_a_real_architecture", num_classes=28)


def test_count_parameters_matches_manual_sum():
    model = build_model("resnet18", num_classes=28)
    counts = count_parameters(model)

    manual_total = sum(p.numel() for p in model.parameters())
    assert counts["total_params"] == manual_total
    # All parameters are trainable immediately after building (no layers
    # frozen), so trainable should equal total for a freshly built model.
    assert counts["trainable_params"] == manual_total


def test_get_device_returns_valid_device():
    device = get_device()
    assert device.type in ("cuda", "cpu")
