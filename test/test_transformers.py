"""Unit tests for custom transformations in fdq.transformers module."""

import unittest
import torch
import numpy as np
from unittest.mock import patch
from omegaconf import OmegaConf
from fdq.transformers import (
    AddValueTransform,
    MultValueTransform,
    DivValueTransform,
    ClampAbsTransform,
    ClampPercTransform,
    ReRangeTransform,
    ReRangeMinMaxTransform,
    Stack3DTransform,
    ResizeMaxDimPadTransform,
    PaddingTransform,
    UnPaddingTransform,
    Float32Transform,
    Uint8Transform,
    Get2DFrom3DTransform,
    SynchronizedRandomVerticalFlip,
    SynchronizedRandomHorizontalFlip,
    RandomGaussianNoiseTransform,
    RandomPoissonNoiseTransform,
    RandomSaltPepperNoiseTransform,
    RandomCutoutTransform,
    RandomBrightnessContrastTransform,
    RandomGammaTransform,
    get_transformer,
)


class TestAddValueTransform(unittest.TestCase):
    """Test AddValueTransform class."""

    def test_add_positive_value(self):
        """Test adding a positive value to tensor."""
        transform = AddValueTransform(5.0)
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([6.0, 7.0, 8.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_add_negative_value(self):
        """Test adding a negative value to tensor."""
        transform = AddValueTransform(-2.5)
        input_tensor = torch.tensor([5.0, 3.0, 1.0])
        expected = torch.tensor([2.5, 0.5, -1.5])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_add_zero(self):
        """Test adding zero to tensor."""
        transform = AddValueTransform(0.0)
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)


class TestMultValueTransform(unittest.TestCase):
    """Test MultValueTransform class."""

    def test_multiply_positive_value(self):
        """Test multiplying by a positive value."""
        transform = MultValueTransform(3.0)
        input_tensor = torch.tensor([1.0, 2.0, 4.0])
        expected = torch.tensor([3.0, 6.0, 12.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_multiply_negative_value(self):
        """Test multiplying by a negative value."""
        transform = MultValueTransform(-2.0)
        input_tensor = torch.tensor([1.0, -2.0, 3.0])
        expected = torch.tensor([-2.0, 4.0, -6.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_multiply_by_zero(self):
        """Test multiplying by zero."""
        transform = MultValueTransform(0.0)
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([0.0, 0.0, 0.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)


class TestDivValueTransform(unittest.TestCase):
    """Test DivValueTransform class."""

    def test_divide_positive_value(self):
        """Test dividing by a positive value."""
        transform = DivValueTransform(2.0)
        input_tensor = torch.tensor([4.0, 6.0, 8.0])
        expected = torch.tensor([2.0, 3.0, 4.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_divide_negative_value(self):
        """Test dividing by a negative value."""
        transform = DivValueTransform(-2.0)
        input_tensor = torch.tensor([4.0, -6.0, 8.0])
        expected = torch.tensor([-2.0, 3.0, -4.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_divide_by_one(self):
        """Test dividing by one."""
        transform = DivValueTransform(1.0)
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)


class TestClampAbsTransform(unittest.TestCase):
    """Test ClampAbsTransform class."""

    def test_clamp_within_bounds(self):
        """Test clamping values within bounds."""
        transform = ClampAbsTransform(0.0, 5.0)
        input_tensor = torch.tensor([1.0, 3.0, 4.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_clamp_above_upper_bound(self):
        """Test clamping values above upper bound."""
        transform = ClampAbsTransform(0.0, 5.0)
        input_tensor = torch.tensor([1.0, 7.0, 10.0])
        expected = torch.tensor([1.0, 5.0, 5.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_clamp_below_lower_bound(self):
        """Test clamping values below lower bound."""
        transform = ClampAbsTransform(2.0, 8.0)
        input_tensor = torch.tensor([0.5, 1.0, 5.0])
        expected = torch.tensor([2.0, 2.0, 5.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)


class TestClampPercTransform(unittest.TestCase):
    """Test ClampPercTransform class."""

    def test_clamp_percentile(self):
        """Test clamping by percentiles."""
        transform = ClampPercTransform(0.2, 0.8)
        # Create tensor where 20th percentile = 2, 80th percentile = 8
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = transform(input_tensor)

        # Values should be clamped to approximately [2.6, 8.4]
        self.assertGreaterEqual(result.min().item(), 2.0)
        self.assertLessEqual(result.max().item(), 9.0)

    def test_clamp_percentile_small_tensor(self):
        """Test clamping percentiles on small tensor."""
        transform = ClampPercTransform(0.25, 0.75)
        input_tensor = torch.tensor([1.0, 5.0, 10.0, 15.0])
        result = transform(input_tensor)

        # Check that clamping occurred
        self.assertEqual(len(result), len(input_tensor))
        self.assertTrue(torch.all(result >= result.min()))
        self.assertTrue(torch.all(result <= result.max()))


class TestReRangeTransform(unittest.TestCase):
    """Test ReRangeTransform class."""

    def test_rerange_basic(self):
        """Test basic re-ranging."""
        transform = ReRangeTransform(0.0, 10.0, 0.0, 1.0)
        input_tensor = torch.tensor([0.0, 5.0, 10.0])
        expected = torch.tensor([0.0, 0.5, 1.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_rerange_negative_to_positive(self):
        """Test re-ranging from negative to positive range."""
        transform = ReRangeTransform(-1.0, 1.0, 0.0, 100.0)
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        expected = torch.tensor([0.0, 50.0, 100.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_rerange_identity(self):
        """Test re-ranging to same range."""
        transform = ReRangeTransform(0.0, 1.0, 0.0, 1.0)
        input_tensor = torch.tensor([0.0, 0.5, 1.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)


class TestReRangeMinMaxTransform(unittest.TestCase):
    """Test ReRangeMinMaxTransform class."""

    def test_rerange_minmax_basic(self):
        """Test basic min-max re-ranging."""
        transform = ReRangeMinMaxTransform(0.0, 1.0)
        input_tensor = torch.tensor([2.0, 4.0, 6.0])
        expected = torch.tensor([0.0, 0.5, 1.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_rerange_minmax_negative_range(self):
        """Test min-max re-ranging to negative range."""
        transform = ReRangeMinMaxTransform(-1.0, 1.0)
        input_tensor = torch.tensor([10.0, 15.0, 20.0])
        expected = torch.tensor([-1.0, 0.0, 1.0])
        result = transform(input_tensor)
        torch.testing.assert_close(result, expected)

    def test_rerange_minmax_constant_input(self):
        """Test min-max re-ranging with constant input."""
        transform = ReRangeMinMaxTransform(0.0, 1.0)
        input_tensor = torch.tensor([5.0, 5.0, 5.0])
        result = transform(input_tensor)

        # When min == max, we get division by zero which results in NaN
        self.assertTrue(torch.isnan(result).all())


class TestStack3DTransform(unittest.TestCase):
    """Test Stack3DTransform class."""

    def test_stack_3d_basic(self):
        """Test basic 3D stacking."""
        transform = Stack3DTransform(stack_n=3)
        input_tensor = torch.ones(2, 4, 4)  # [C, H, W]
        result = transform(input_tensor)

        # Stack3DTransform stacks along dim=2, so [C, H, W] -> [C, D, H, W]
        # torch.stack([t] * stack_n, dim=2) gives [2, 4, 3, 4]
        self.assertEqual(result.shape, (2, 4, 3, 4))  # [C, H, D, W]

    def test_stack_3d_single_stack(self):
        """Test 3D stacking with single stack."""
        transform = Stack3DTransform(stack_n=1)
        input_tensor = torch.ones(1, 2, 2)  # [C, H, W]
        result = transform(input_tensor)

        # Single stack: [C, H, W] -> [C, H, D, W] where D=1
        self.assertEqual(result.shape, (1, 2, 1, 2))


class TestResizeMaxDimPadTransform(unittest.TestCase):
    """Test ResizeMaxDimPadTransform class."""

    def test_resize_max_dim_square_input(self):
        """Test resizing with square input."""
        transform = ResizeMaxDimPadTransform(64)
        input_tensor = torch.rand(3, 32, 32)
        result = transform(input_tensor)

        self.assertEqual(result.shape, (3, 64, 64))

    def test_resize_max_dim_rectangular_input(self):
        """Test resizing with rectangular input."""
        transform = ResizeMaxDimPadTransform(100)
        input_tensor = torch.rand(3, 50, 80)  # H < W
        result = transform(input_tensor)

        self.assertEqual(result.shape, (3, 100, 100))

    def test_resize_max_dim_different_modes(self):
        """Test resizing with different padding modes."""
        transform_constant = ResizeMaxDimPadTransform(64, mode="constant", value=0.5)
        transform_replicate = ResizeMaxDimPadTransform(64, mode="replicate")

        input_tensor = torch.rand(1, 30, 40)

        result_constant = transform_constant(input_tensor)
        result_replicate = transform_replicate(input_tensor)

        self.assertEqual(result_constant.shape, (1, 64, 64))
        self.assertEqual(result_replicate.shape, (1, 64, 64))


class TestPaddingTransform(unittest.TestCase):
    """Test PaddingTransform class."""

    def test_padding_2d(self):
        """Test 2D padding."""
        transform = PaddingTransform((1, 1, 2, 2))  # pad last 2 dims
        input_tensor = torch.ones(3, 4)
        result = transform(input_tensor)

        # Padding (1,1,2,2): last dim +2, second-to-last dim +4
        self.assertEqual(result.shape, (3 + 2 + 2, 4 + 1 + 1))  # (7, 6)

    def test_padding_constant_mode(self):
        """Test padding with constant mode."""
        transform = PaddingTransform((1, 1), padding_mode="constant", padding_value=5.0)
        input_tensor = torch.ones(3, 4)
        result = transform(input_tensor)

        # Check shape
        self.assertEqual(result.shape, (3, 6))

        # Check padding values
        self.assertTrue(torch.allclose(result[:, 0], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(result[:, -1], torch.tensor(5.0)))

    def test_padding_invalid_mode(self):
        """Test padding with invalid mode."""
        with self.assertRaises(ValueError):
            PaddingTransform((1, 1), padding_mode="invalid")


class TestUnPaddingTransform(unittest.TestCase):
    """Test UnPaddingTransform class."""

    def test_unpadding_4d(self):
        """Test unpadding 4D tensor."""
        # First pad, then unpad
        pad_transform = PaddingTransform((1, 1, 2, 2))
        unpad_transform = UnPaddingTransform((1, 1, 2, 2))

        input_tensor = torch.rand(1, 3, 4, 4)
        padded = pad_transform(input_tensor)
        unpadded = unpad_transform(padded)

        torch.testing.assert_close(unpadded, input_tensor)

    def test_unpadding_5d(self):
        """Test unpadding 5D tensor."""
        unpad_transform = UnPaddingTransform((1, 1, 1, 1))
        input_tensor = torch.rand(2, 3, 4, 6, 8)
        result = unpad_transform(input_tensor)

        # Padding (1, 1, 1, 1) removes:
        # - 1 from each side of last dim: 8 - 1 - 1 = 6
        # - 1 from each side of second-to-last dim: 6 - 1 - 1 = 4
        self.assertEqual(result.shape, (2, 3, 4, 4, 6))  # reduced by padding

    def test_unpadding_3d(self):
        """Test unpadding 3D tensor."""
        unpad_transform = UnPaddingTransform((1, 1))
        input_tensor = torch.rand(2, 3, 4)  # 3D tensor
        result = unpad_transform(input_tensor)

        # Padding (1, 1) removes 1 from each side of last dim: 4 - 1 - 1 = 2
        self.assertEqual(result.shape, (2, 3, 2))


class TestFloat32Transform(unittest.TestCase):
    """Test Float32Transform class."""

    def test_convert_to_float32(self):
        """Test conversion to float32."""
        transform = Float32Transform()
        input_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = transform(input_tensor)

        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_float32_already_float32(self):
        """Test conversion when already float32."""
        transform = Float32Transform()
        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = transform(input_tensor)

        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, input_tensor)


class TestUint8Transform(unittest.TestCase):
    """Test Uint8Transform class."""

    def test_convert_to_uint8(self):
        """Test conversion to uint8."""
        transform = Uint8Transform()
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = transform(input_tensor)

        self.assertEqual(result.dtype, torch.uint8)
        torch.testing.assert_close(result, torch.tensor([1, 2, 3], dtype=torch.uint8))

    def test_uint8_clipping(self):
        """Test uint8 conversion with clipping."""
        transform = Uint8Transform()
        input_tensor = torch.tensor([-1.0, 256.0, 128.0])
        result = transform(input_tensor)

        self.assertEqual(result.dtype, torch.uint8)
        # Values should be clipped to uint8 range [0, 255]


class TestGet2DFrom3DTransform(unittest.TestCase):
    """Test Get2DFrom3DTransform class."""

    def test_get_2d_middle_slice(self):
        """Test getting middle slice from 3D tensor."""
        input_tensor = torch.rand(6, 4, 4)
        transform = Get2DFrom3DTransform(axis=0, index=None)  # None = middle
        result = transform(input_tensor)

        self.assertEqual(result.shape, (4, 4))
        # Middle of 6 elements is index 3 (6//2 = 3)
        torch.testing.assert_close(result, input_tensor[3, :, :])  # middle slice

    def test_get_2d_specific_index(self):
        """Test getting specific slice from 3D tensor."""
        transform = Get2DFrom3DTransform(axis=1, index=2)
        input_tensor = torch.rand(3, 5, 4)
        result = transform(input_tensor)

        self.assertEqual(result.shape, (3, 4))
        torch.testing.assert_close(result, input_tensor[:, 2, :])

    def test_get_2d_invalid_axis(self):
        """Test invalid axis for 3D tensor."""
        transform = Get2DFrom3DTransform(axis=3, index=0)
        input_tensor = torch.rand(2, 3, 4)  # 3D tensor

        with self.assertRaises(ValueError):
            transform(input_tensor)

    def test_get_2d_negative_axis(self):
        """Test negative axis for 3D tensor."""
        transform = Get2DFrom3DTransform(axis=-1, index=0)
        input_tensor = torch.rand(2, 3, 4)

        with self.assertRaises(ValueError):
            transform(input_tensor)

    def test_get_2d_middle_slice_recomputed_for_smaller_tensor(self):
        """A shared instance must not cache the first call's middle index.

        Transform instances are built once and reused for every tensor passed
        through them over an experiment's lifetime. If the same instance is later
        called with a smaller tensor along `axis` (e.g. a downsampled latent after
        a full-resolution volume), the middle slice must be recomputed rather than
        reusing the previous call's index.
        """
        transform = Get2DFrom3DTransform(axis=0, index=None)

        large_tensor = torch.rand(64, 4, 4)
        large_result = transform(large_tensor)
        torch.testing.assert_close(large_result, large_tensor[32, :, :])

        small_tensor = torch.rand(16, 4, 4)
        small_result = transform(small_tensor)
        torch.testing.assert_close(small_result, small_tensor[8, :, :])


class TestSynchronizedRandomVerticalFlip(unittest.TestCase):
    """Test SynchronizedRandomVerticalFlip class."""

    def test_synchronized_vertical_flip_deterministic(self):
        """Test that synchronized flip affects all tensors the same way."""
        transform = SynchronizedRandomVerticalFlip(p=1.0)  # Always flip

        tensor1 = torch.arange(12).reshape(3, 4).float()
        tensor2 = torch.arange(12, 24).reshape(3, 4).float()

        # Mock the random generator to ensure deterministic behavior
        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.3])  # < p=1.0, so flip

            result1, result2 = transform(tensor1, tensor2)

        # Both tensors should be flipped vertically (dim=-2)
        expected1 = torch.flip(tensor1, dims=[-2])
        expected2 = torch.flip(tensor2, dims=[-2])

        torch.testing.assert_close(result1, expected1)
        torch.testing.assert_close(result2, expected2)

    def test_synchronized_vertical_flip_no_flip(self):
        """Test that synchronized flip returns original tensors when no flip."""
        transform = SynchronizedRandomVerticalFlip(p=0.0)  # Never flip

        tensor1 = torch.rand(2, 3, 4)
        tensor2 = torch.rand(2, 3, 4)

        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.8])  # > p=0.0, so no flip

            result1, result2 = transform(tensor1, tensor2)

        torch.testing.assert_close(result1, tensor1)
        torch.testing.assert_close(result2, tensor2)

    def test_synchronized_vertical_flip_single_tensor(self):
        """Test synchronized flip with single tensor."""
        transform = SynchronizedRandomVerticalFlip(p=1.0)
        tensor = torch.arange(6).reshape(2, 3).float()

        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.3])  # < p=1.0, so flip

            result = transform(tensor)

        expected = torch.flip(tensor, dims=[-2])
        torch.testing.assert_close(result[0], expected)

    def test_apply_transform_method(self):
        """Test the apply_transform method."""
        transform = SynchronizedRandomVerticalFlip(p=1.0)
        tensor1 = torch.rand(2, 3)
        tensor2 = torch.rand(2, 3)

        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.3])  # < p=1.0, so flip

            result = transform.apply_transform((tensor1, tensor2))

        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0], torch.flip(tensor1, dims=[-2]))
        torch.testing.assert_close(result[1], torch.flip(tensor2, dims=[-2]))

    def test_generator_deterministic_behavior(self):
        """Test that generator parameter provides deterministic behavior."""
        # Create a fixed generator for deterministic results
        generator = torch.Generator()
        generator.manual_seed(42)

        transform = SynchronizedRandomVerticalFlip(p=0.5, generator=generator)
        tensor1 = torch.arange(12).reshape(3, 4).float()
        tensor2 = torch.arange(12, 24).reshape(3, 4).float()

        # First call
        result1_1, result1_2 = transform(tensor1, tensor2)

        # Reset generator to same seed
        generator.manual_seed(42)

        # Second call should produce identical results
        result2_1, result2_2 = transform(tensor1, tensor2)

        torch.testing.assert_close(result1_1, result2_1)
        torch.testing.assert_close(result1_2, result2_2)

    def test_generator_different_seeds(self):
        """Test that different generator seeds produce different results."""
        tensor1 = torch.arange(12).reshape(3, 4).float()
        tensor2 = torch.arange(12, 24).reshape(3, 4).float()

        # Generator with seed 42
        generator1 = torch.Generator()
        generator1.manual_seed(42)
        transform1 = SynchronizedRandomVerticalFlip(p=0.5, generator=generator1)

        # Generator with seed 123
        generator2 = torch.Generator()
        generator2.manual_seed(123)
        transform2 = SynchronizedRandomVerticalFlip(p=0.5, generator=generator2)

        # Run multiple times to increase chance of different outcomes
        results_different = False
        for _ in range(10):
            generator1.manual_seed(42)
            generator2.manual_seed(123)

            result1_1, result1_2 = transform1(tensor1, tensor2)
            result2_1, result2_2 = transform2(tensor1, tensor2)

            # Check if results are different (at least one should be different across iterations)
            if not torch.equal(result1_1, result2_1) or not torch.equal(result1_2, result2_2):
                results_different = True
                break

        # With different seeds, we should get different results at least once
        self.assertTrue(results_different, "Different generator seeds should produce different results")

    def test_generator_none_uses_global_state(self):
        """Test that generator=None uses global random state."""
        transform = SynchronizedRandomVerticalFlip(p=0.5, generator=None)
        tensor = torch.arange(6).reshape(2, 3).float()

        # This should work without error (uses global random state)
        result = transform(tensor)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, tensor.shape)


class TestSynchronizedRandomHorizontalFlip(unittest.TestCase):
    """Test SynchronizedRandomHorizontalFlip class."""

    def test_synchronized_horizontal_flip_deterministic(self):
        """Test that synchronized flip affects all tensors the same way."""
        transform = SynchronizedRandomHorizontalFlip(p=1.0)  # Always flip

        tensor1 = torch.arange(12).reshape(3, 4).float()
        tensor2 = torch.arange(12, 24).reshape(3, 4).float()

        # Mock the random generator to ensure deterministic behavior
        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.3])  # < p=1.0, so flip

            result1, result2 = transform(tensor1, tensor2)

        # Both tensors should be flipped horizontally (dim=-1)
        expected1 = torch.flip(tensor1, dims=[-1])
        expected2 = torch.flip(tensor2, dims=[-1])

        torch.testing.assert_close(result1, expected1)
        torch.testing.assert_close(result2, expected2)

    def test_synchronized_horizontal_flip_no_flip(self):
        """Test that synchronized flip returns original tensors when no flip."""
        transform = SynchronizedRandomHorizontalFlip(p=0.0)  # Never flip

        tensor1 = torch.rand(2, 3, 4)
        tensor2 = torch.rand(2, 3, 4)

        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.8])  # > p=0.0, so no flip

            result1, result2 = transform(tensor1, tensor2)

        torch.testing.assert_close(result1, tensor1)
        torch.testing.assert_close(result2, tensor2)

    def test_multiple_tensors_horizontal_flip(self):
        """Test horizontal flip with multiple tensors."""
        transform = SynchronizedRandomHorizontalFlip(p=1.0)

        tensors = [torch.rand(2, 4) for _ in range(5)]

        with patch("torch.randint") as mock_randint, patch("torch.rand") as mock_rand:
            mock_randint.return_value = torch.tensor([42])
            mock_rand.return_value = torch.tensor([0.3])  # < p=1.0, so flip

            results = transform(*tensors)

        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            expected = torch.flip(tensors[i], dims=[-1])
            torch.testing.assert_close(result, expected)

    def test_generator_deterministic_behavior(self):
        """Test that generator parameter provides deterministic behavior."""
        # Create a fixed generator for deterministic results
        generator = torch.Generator()
        generator.manual_seed(42)

        transform = SynchronizedRandomHorizontalFlip(p=0.5, generator=generator)
        tensor1 = torch.arange(12).reshape(3, 4).float()
        tensor2 = torch.arange(12, 24).reshape(3, 4).float()

        # First call
        result1_1, result1_2 = transform(tensor1, tensor2)

        # Reset generator to same seed
        generator.manual_seed(42)

        # Second call should produce identical results
        result2_1, result2_2 = transform(tensor1, tensor2)

        torch.testing.assert_close(result1_1, result2_1)
        torch.testing.assert_close(result1_2, result2_2)

    def test_generator_different_seeds(self):
        """Test that different generator seeds produce different results."""
        tensor1 = torch.arange(12).reshape(3, 4).float()
        tensor2 = torch.arange(12, 24).reshape(3, 4).float()

        # Generator with seed 42
        generator1 = torch.Generator()
        generator1.manual_seed(42)
        transform1 = SynchronizedRandomHorizontalFlip(p=0.5, generator=generator1)

        # Generator with seed 123
        generator2 = torch.Generator()
        generator2.manual_seed(123)
        transform2 = SynchronizedRandomHorizontalFlip(p=0.5, generator=generator2)

        # Run multiple times to increase chance of different outcomes
        results_different = False
        for _ in range(10):
            generator1.manual_seed(42)
            generator2.manual_seed(123)

            result1_1, result1_2 = transform1(tensor1, tensor2)
            result2_1, result2_2 = transform2(tensor1, tensor2)

            # Check if results are different (at least one should be different across iterations)
            if not torch.equal(result1_1, result2_1) or not torch.equal(result1_2, result2_2):
                results_different = True
                break

        # With different seeds, we should get different results at least once
        self.assertTrue(results_different, "Different generator seeds should produce different results")

    def test_generator_none_uses_global_state(self):
        """Test that generator=None uses global random state."""
        transform = SynchronizedRandomHorizontalFlip(p=0.5, generator=None)
        tensor = torch.arange(6).reshape(2, 3).float()

        # This should work without error (uses global random state)
        result = transform(tensor)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, tensor.shape)


class TestRandomGaussianNoiseTransform(unittest.TestCase):
    """Test RandomGaussianNoiseTransform class."""

    def test_zero_sigma_is_identity(self):
        """Test that a sigma range of exactly 0 leaves the tensor unchanged."""
        transform = RandomGaussianNoiseTransform(sigma_min=0.0, sigma_max=0.0, p=1.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_probability_zero_no_op(self):
        """Test that p=0.0 never applies the noise."""
        transform = RandomGaussianNoiseTransform(sigma_min=1.0, sigma_max=2.0, p=0.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_noise_changes_values_and_preserves_shape(self):
        """Test that noise is added and the tensor shape/dtype are preserved."""
        transform = RandomGaussianNoiseTransform(sigma_min=0.5, sigma_max=0.5, p=1.0)
        input_tensor = torch.zeros(4, 8, 8)
        result = transform(input_tensor)

        self.assertEqual(result.shape, input_tensor.shape)
        self.assertEqual(result.dtype, input_tensor.dtype)
        self.assertFalse(torch.equal(result, input_tensor))

    def test_works_on_3d_volume(self):
        """Test that the transform works on a (C, D, H, W) shaped volume."""
        transform = RandomGaussianNoiseTransform(sigma_min=0.2, sigma_max=0.2, p=1.0)
        input_tensor = torch.zeros(2, 6, 10, 10)
        result = transform(input_tensor)

        self.assertEqual(result.shape, input_tensor.shape)
        self.assertFalse(torch.equal(result, input_tensor))

    def test_invalid_sigma_range_raises(self):
        """Test that an invalid sigma range raises ValueError."""
        with self.assertRaises(ValueError):
            RandomGaussianNoiseTransform(sigma_min=0.5, sigma_max=0.1)
        with self.assertRaises(ValueError):
            RandomGaussianNoiseTransform(sigma_min=-1.0, sigma_max=0.1)

    def test_generator_deterministic_behavior(self):
        """Test that a fixed generator seed reproduces the same noise."""
        input_tensor = torch.zeros(3, 8, 8)

        generator1 = torch.Generator().manual_seed(42)
        transform1 = RandomGaussianNoiseTransform(sigma_min=0.1, sigma_max=0.5, p=1.0, generator=generator1)
        result1 = transform1(input_tensor)

        generator2 = torch.Generator().manual_seed(42)
        transform2 = RandomGaussianNoiseTransform(sigma_min=0.1, sigma_max=0.5, p=1.0, generator=generator2)
        result2 = transform2(input_tensor)

        torch.testing.assert_close(result1, result2)


class TestRandomPoissonNoiseTransform(unittest.TestCase):
    """Test RandomPoissonNoiseTransform class."""

    def test_probability_zero_no_op(self):
        """Test that p=0.0 never applies the noise."""
        transform = RandomPoissonNoiseTransform(p=0.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_output_shape_dtype_and_nonnegative(self):
        """Test that shape/dtype are preserved and the noisy output stays non-negative."""
        transform = RandomPoissonNoiseTransform(scale_min=0.5, scale_max=0.5, p=1.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)

        self.assertEqual(result.shape, input_tensor.shape)
        self.assertEqual(result.dtype, input_tensor.dtype)
        self.assertTrue(torch.all(result >= 0))

    def test_works_on_3d_volume(self):
        """Test that the transform works on a (C, D, H, W) shaped volume."""
        transform = RandomPoissonNoiseTransform(scale_min=1.0, scale_max=1.0, p=1.0)
        input_tensor = torch.rand(2, 6, 10, 10)
        result = transform(input_tensor)
        self.assertEqual(result.shape, input_tensor.shape)

    def test_invalid_scale_range_raises(self):
        """Test that an invalid scale range raises ValueError."""
        with self.assertRaises(ValueError):
            RandomPoissonNoiseTransform(scale_min=0.0, scale_max=1.0)
        with self.assertRaises(ValueError):
            RandomPoissonNoiseTransform(scale_min=0.9, scale_max=0.1)


class TestRandomSaltPepperNoiseTransform(unittest.TestCase):
    """Test RandomSaltPepperNoiseTransform class."""

    def test_probability_zero_no_op(self):
        """Test that p=0.0 never applies the noise."""
        transform = RandomSaltPepperNoiseTransform(amount=0.5, p=0.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_amount_zero_is_identity(self):
        """Test that amount=0.0 leaves the tensor unchanged."""
        transform = RandomSaltPepperNoiseTransform(amount=0.0, p=1.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_corrupted_fraction_matches_amount_approximately(self):
        """Test that roughly `amount` of the elements are set to salt/pepper values."""
        transform = RandomSaltPepperNoiseTransform(
            amount=0.3, salt_vs_pepper=0.5, salt_value=9.0, pepper_value=-9.0, p=1.0
        )
        input_tensor = torch.rand(200, 200)
        result = transform(input_tensor)

        corrupted_frac = ((result == 9.0) | (result == -9.0)).float().mean().item()
        self.assertAlmostEqual(corrupted_frac, 0.3, delta=0.02)

    def test_works_on_3d_volume(self):
        """Test that the transform works on a (C, D, H, W) shaped volume."""
        transform = RandomSaltPepperNoiseTransform(amount=0.1, p=1.0)
        input_tensor = torch.rand(2, 6, 10, 10)
        result = transform(input_tensor)
        self.assertEqual(result.shape, input_tensor.shape)

    def test_invalid_amount_raises(self):
        """Test that an out-of-range amount raises ValueError."""
        with self.assertRaises(ValueError):
            RandomSaltPepperNoiseTransform(amount=1.5)
        with self.assertRaises(ValueError):
            RandomSaltPepperNoiseTransform(salt_vs_pepper=-0.1)


class TestRandomCutoutTransform(unittest.TestCase):
    """Test RandomCutoutTransform class."""

    def test_probability_zero_no_op(self):
        """Test that p=0.0 never applies the cutout."""
        transform = RandomCutoutTransform(p=0.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_cutout_2d_writes_fill_value(self):
        """Test that a 2D cutout writes the fill value and preserves shape."""
        transform = RandomCutoutTransform(
            n_spatial_dims=2, num_patches=1, min_frac=0.5, max_frac=0.5, fill_value=-1.0, p=1.0
        )
        input_tensor = torch.rand(3, 20, 20)
        result = transform(input_tensor)

        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue(torch.any(result == -1.0))
        # channel dim must not be touched by the spatial cutout
        self.assertFalse(torch.all(result == -1.0))

    def test_cutout_3d_volume(self):
        """Test that the cutout works on a (C, D, H, W) shaped volume, masking D/H/W."""
        transform = RandomCutoutTransform(
            n_spatial_dims=3, num_patches=1, min_frac=0.5, max_frac=0.5, fill_value=-1.0, p=1.0
        )
        input_tensor = torch.rand(2, 10, 10, 10)
        result = transform(input_tensor)

        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue(torch.any(result == -1.0))

    def test_multiple_patches(self):
        """Test that requesting multiple patches can cover a larger fraction of the tensor."""
        transform = RandomCutoutTransform(
            n_spatial_dims=2, num_patches=5, min_frac=0.1, max_frac=0.2, fill_value=0.0, p=1.0
        )
        input_tensor = torch.ones(1, 50, 50)
        result = transform(input_tensor)
        self.assertTrue(torch.any(result == 0.0))

    def test_invalid_n_spatial_dims_raises(self):
        """Test that requesting more spatial dims than the tensor has raises ValueError."""
        transform = RandomCutoutTransform(n_spatial_dims=3, p=1.0)
        input_tensor = torch.rand(4, 4)
        with self.assertRaises(ValueError):
            transform(input_tensor)

    def test_invalid_frac_range_raises(self):
        """Test that an invalid fraction range raises ValueError."""
        with self.assertRaises(ValueError):
            RandomCutoutTransform(min_frac=0.6, max_frac=0.5)
        with self.assertRaises(ValueError):
            RandomCutoutTransform(min_frac=0.0)


class TestRandomBrightnessContrastTransform(unittest.TestCase):
    """Test RandomBrightnessContrastTransform class."""

    def test_probability_zero_no_op(self):
        """Test that p=0.0 never applies the jitter."""
        transform = RandomBrightnessContrastTransform(brightness=0.5, contrast=0.5, p=0.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_zero_jitter_is_identity(self):
        """Test that brightness=0 and contrast=0 leave the tensor unchanged."""
        transform = RandomBrightnessContrastTransform(brightness=0.0, contrast=0.0, p=1.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_jitter_changes_values_and_preserves_shape(self):
        """Test that jitter changes values while preserving shape."""
        transform = RandomBrightnessContrastTransform(brightness=0.5, contrast=0.5, p=1.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)

        self.assertEqual(result.shape, input_tensor.shape)
        self.assertFalse(torch.equal(result, input_tensor))

    def test_works_on_3d_volume(self):
        """Test that the transform works on a (C, D, H, W) shaped volume."""
        transform = RandomBrightnessContrastTransform(brightness=0.3, contrast=0.3, p=1.0)
        input_tensor = torch.rand(2, 6, 10, 10)
        result = transform(input_tensor)
        self.assertEqual(result.shape, input_tensor.shape)

    def test_invalid_params_raise(self):
        """Test that negative brightness/contrast raise ValueError."""
        with self.assertRaises(ValueError):
            RandomBrightnessContrastTransform(brightness=-0.1)
        with self.assertRaises(ValueError):
            RandomBrightnessContrastTransform(contrast=-0.1)


class TestRandomGammaTransform(unittest.TestCase):
    """Test RandomGammaTransform class."""

    def test_probability_zero_no_op(self):
        """Test that p=0.0 never applies the gamma correction."""
        transform = RandomGammaTransform(gamma_min=2.0, gamma_max=3.0, p=0.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_gamma_one_is_identity(self):
        """Test that gamma=1.0 leaves a non-negative tensor unchanged."""
        transform = RandomGammaTransform(gamma_min=1.0, gamma_max=1.0, p=1.0)
        input_tensor = torch.rand(3, 16, 16)
        result = transform(input_tensor)
        torch.testing.assert_close(result, input_tensor)

    def test_gamma_preserves_sign(self):
        """Test that gamma correction preserves the sign of each element."""
        transform = RandomGammaTransform(gamma_min=2.0, gamma_max=2.0, p=1.0)
        input_tensor = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = transform(input_tensor)
        expected = torch.tensor([-4.0, -0.25, 0.0, 0.25, 4.0])
        torch.testing.assert_close(result, expected)

    def test_works_on_3d_volume(self):
        """Test that the transform works on a (C, D, H, W) shaped volume."""
        transform = RandomGammaTransform(gamma_min=0.8, gamma_max=1.2, p=1.0)
        input_tensor = torch.rand(2, 6, 10, 10)
        result = transform(input_tensor)
        self.assertEqual(result.shape, input_tensor.shape)

    def test_invalid_gamma_range_raises(self):
        """Test that an invalid gamma range raises ValueError."""
        with self.assertRaises(ValueError):
            RandomGammaTransform(gamma_min=0.0, gamma_max=1.0)
        with self.assertRaises(ValueError):
            RandomGammaTransform(gamma_min=1.5, gamma_max=1.0)


class TestNewAugmentationDispatch(unittest.TestCase):
    """Test config-based dispatch (get_transformer) for the new augmentation transforms."""

    def test_dispatch_creates_expected_types(self):
        """Test that each new transformer name resolves to the correct class."""
        cases = [
            ("RandomGaussianNoise", {"sigma_min": 0.0, "sigma_max": 0.1}, RandomGaussianNoiseTransform),
            ("RandomPoissonNoise", {"scale_min": 0.5, "scale_max": 1.0}, RandomPoissonNoiseTransform),
            ("RandomSaltPepperNoise", {"amount": 0.05}, RandomSaltPepperNoiseTransform),
            ("RandomCutout", {"n_spatial_dims": 3}, RandomCutoutTransform),
            ("RandomBrightnessContrast", {"brightness": 0.1}, RandomBrightnessContrastTransform),
            ("RandomGamma", {"gamma_min": 0.8, "gamma_max": 1.2}, RandomGammaTransform),
        ]
        for name, params, expected_cls in cases:
            cfg = OmegaConf.create({name: params})
            transformer = get_transformer(cfg)
            self.assertIsInstance(transformer, expected_cls)

    def test_dispatch_without_parameters_uses_defaults(self):
        """Test that bare string transform names fall back to default parameters."""
        transformer = get_transformer("RandomCutout")
        self.assertIsInstance(transformer, RandomCutoutTransform)

    def test_dispatch_transformer_applies_to_2d_and_3d(self):
        """Test that a dispatched transformer can be applied to both 2D and 3D tensors."""
        cfg = OmegaConf.create({"RandomCutout": {"n_spatial_dims": 3, "p": 1.0, "fill_value": -1.0}})
        transformer = get_transformer(cfg)

        volume = torch.rand(2, 8, 12, 12)
        result = transformer(volume)
        self.assertEqual(result.shape, volume.shape)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_tensor(self):
        """Test transforms with empty tensors."""
        empty_tensor = torch.empty(0, 3, 4)

        # Test transforms that should work with empty tensors
        add_transform = AddValueTransform(1.0)
        result = add_transform(empty_tensor)
        self.assertEqual(result.shape, empty_tensor.shape)

    def test_single_element_tensor(self):
        """Test transforms with single element tensors."""
        single_tensor = torch.tensor([5.0])

        mult_transform = MultValueTransform(2.0)
        result = mult_transform(single_tensor)
        torch.testing.assert_close(result, torch.tensor([10.0]))

    def test_large_tensor_clamp_perc(self):
        """Test ClampPercTransform with large tensor (triggers rdm_reduce)."""
        transform = ClampPercTransform(0.1, 0.9)
        # Create a tensor larger than the threshold (12582912 elements)
        large_tensor = torch.randn(5000, 5000)  # 25M elements

        result = transform(large_tensor)
        self.assertEqual(result.shape, large_tensor.shape)

    def test_probability_edge_cases(self):
        """Test synchronized flips with edge probability cases."""
        # Test p=0.0 (never flip)
        transform_never = SynchronizedRandomVerticalFlip(p=0.0)
        tensor = torch.rand(2, 3)

        with patch("torch.rand", return_value=torch.tensor([0.5])):
            result = transform_never(tensor)
            torch.testing.assert_close(result[0], tensor)

        # Test p=1.0 (always flip)
        transform_always = SynchronizedRandomVerticalFlip(p=1.0)

        with patch("torch.rand", return_value=torch.tensor([0.5])):
            result = transform_always(tensor)
            expected = torch.flip(tensor, dims=[-2])
            torch.testing.assert_close(result[0], expected)


if __name__ == "__main__":
    unittest.main()
