import pytest
import numpy as np
from project import apply_grayscale, apply_inversion, apply_sepia


def dummy_image():
    """Creates a random 10x10 RGB image array for testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

def test_apply_grayscale(dummy_image):

    result = apply_grayscale(dummy_image)
    
    assert result.shape == dummy_image.shape 
    assert np.array_equal(result[:, :, 0], result[:, :, 1]) 
    assert np.array_equal(result[:, :, 1], result[:, :, 2]) 

def test_apply_inversion(dummy_image):
        result = apply_inversion(dummy_image)
    
    expected = 255 - dummy_image
    assert np.array_equal(result, expected)
    
    double_inverted = apply_inversion(result)
    assert np.array_equal(double_inverted, dummy_image)

def test_apply_sepia(dummy_image):
    """Verify that apply_sepia Works."""
    # Force some pure white pixels to test the 255 clipping boundary
    dummy_image[0, 0] = [255, 255, 255] 
    result = apply_sepia(dummy_image)
    
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8
    assert np.all(result >= 0) and np.all(result <= 255)