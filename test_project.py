import pytest
import numpy as np
from project import apply_grayscale, apply_inversion, apply_sepia


def test_apply_grayscale():
    """Verify that apply_grayscale returns a 3-channel RGB array whose R, G and B
    channels are identical, i.e. the image is visually gray."""
    pass


def test_apply_inversion():
    """Verify that apply_inversion returns each pixel value subtracted from 255,
    and that inverting twice restores the original array."""
    pass


def test_apply_sepia():
    """Verify that apply_sepia returns a uint8 array of the same shape as its input,
    with all values clipped to the 0-255 range."""
    pass
