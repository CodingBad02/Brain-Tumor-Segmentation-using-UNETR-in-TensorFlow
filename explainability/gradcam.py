"""
Grad-CAM Implementation for UNETR Brain Tumor Segmentation
==========================================================

This module provides Grad-CAM (Gradient-weighted Class Activation Mapping)
functionality specifically designed for the UNETR 2D architecture used in
brain tumor segmentation.

The implementation handles the unique architecture of UNETR with its
transformer encoder and CNN decoder, focusing on the decoder convolutional
layers that preserve spatial information for visualization.
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


class GradCAM:
    """
    Grad-CAM implementation for UNETR 2D model

    This class generates gradient-weighted class activation maps to visualize
    which regions of the input image contribute most to the segmentation
    decisions.
    """

    def __init__(self, model: tf.keras.Model, target_layer_names: Union[str, List[str]],
                 input_shape: Tuple[int, ...] = (256, 256, 3)):
        """
        Initialize Grad-CAM for UNETR model

        Args:
            model: Loaded UNETR model
            target_layer_names: Name(s) of target convolutional layer(s) for Grad-CAM
            input_shape: Input image shape (H, W, C)
        """
        self.model = model
        self.input_shape = input_shape

        # Handle single layer name or list of layer names
        if isinstance(target_layer_names, str):
            self.target_layer_names = [target_layer_names]
        else:
            self.target_layer_names = target_layer_names

        # Verify target layers exist
        self._verify_target_layers()

        # Storage for gradients and activations
        self.gradients = {}
        self.activations = {}

        # Register hooks
        self._register_hooks()

    def _verify_target_layers(self):
        """Verify that target layers exist in the model"""
        model_layers = [layer.name for layer in self.model.layers]

        for layer_name in self.target_layer_names:
            if layer_name not in model_layers:
                available_conv_layers = [
                    layer.name for layer in self.model.layers
                    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose))
                ]
                raise ValueError(
                    f"Target layer '{layer_name}' not found in model. "
                    f"Available conv layers: {available_conv_layers[:10]}..."
                )

    def _register_hooks(self):
        """Register hooks for target layers by modifying the model"""
        # Since TensorFlow doesn't have PyTorch-style hooks,
        # we'll capture activations during forward pass using GradientTape
        pass

    def _compute_gradients(self, input_image: np.ndarray, target_class_idx: int = 0) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Compute gradients for target class and capture activations

        Args:
            input_image: Input image tensor [1, H, W, C]
            target_class_idx: Target class index (for binary segmentation, use 0)

        Returns:
            Tuple of (gradients, activations) dictionaries per target layer
        """
        input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)

        # Clear previous activations
        self.activations.clear()

        # Create a model that outputs intermediate layers
        layer_outputs = [self.model.get_layer(name).output for name in self.target_layer_names]
        gradcam_model = tf.keras.Model(inputs=self.model.input,
                                     outputs=[self.model.output] + layer_outputs)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_tensor)

            # Forward pass to get predictions and intermediate activations
            outputs = gradcam_model(input_tensor, training=False)
            predictions = outputs[0]
            activations_list = outputs[1:]

            # Store activations
            for layer_name, activation in zip(self.target_layer_names, activations_list):
                self.activations[layer_name] = activation

            # For segmentation, we use the mean prediction value as target
            if len(predictions.shape) == 4:  # [B, H, W, C]
                # Use spatial average of the positive predictions
                target_output = tf.reduce_mean(predictions[:, :, :, target_class_idx])
            else:
                target_output = tf.reduce_mean(predictions)

        # Compute gradients for each target layer
        gradients = {}
        for layer_name, activation in zip(self.target_layer_names, activations_list):
            gradients[layer_name] = tape.gradient(target_output, activation)

        return gradients, self.activations

    def generate_heatmap(self, input_image: np.ndarray,
                        target_class_idx: int = 0,
                        aggregation_method: str = 'mean',
                        resize_to_input: bool = True) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image

        Args:
            input_image: Input image [H, W, C] or [1, H, W, C]
            target_class_idx: Target class index
            aggregation_method: How to aggregate multiple layers ('mean', 'max', 'sum', 'weighted')
            resize_to_input: Whether to resize heatmap to input size

        Returns:
            Grad-CAM heatmap [H, W]
        """
        # Ensure input has batch dimension
        if len(input_image.shape) == 3:
            input_image = np.expand_dims(input_image, axis=0)

        # UNETR needs flattened patches as input
        # Store original image shape for later
        original_shape = input_image.shape[1:3]  # (H, W)

        # Convert image to patches (same preprocessing as in test.py)
        from patchify import patchify
        patch_size = 16
        num_channels = 3

        x = input_image[0]  # Remove batch dimension
        patch_shape = (patch_size, patch_size, num_channels)
        patches = patchify(x, patch_shape, patch_size)

        # Flatten patches
        num_patches = 256
        flat_patches_shape = (num_patches, patch_size * patch_size * num_channels)
        patches = np.reshape(patches, flat_patches_shape)
        patches = patches.astype(np.float32)
        patches = np.expand_dims(patches, axis=0)  # Add batch dimension

        # Use patches as model input
        model_input = patches

        # Clear previous gradients
        self.gradients.clear()

        # Compute gradients and capture activations
        gradients, self.activations = self._compute_gradients(model_input, target_class_idx)

        # Process each layer
        layer_heatmaps = []
        for layer_name in self.target_layer_names:
            if layer_name in self.activations and layer_name in gradients:
                activations = self.activations[layer_name].numpy()
                grads = gradients[layer_name].numpy()

                # Calculate importance weights (global average pooling of gradients)
                if len(grads.shape) == 4:  # [B, H, W, C]
                    weights = np.mean(grads, axis=(1, 2))  # [B, C]
                else:
                    continue

                # Weighted combination of activation maps
                if len(activations.shape) == 4:  # [B, H, W, C]
                    # Create heatmap
                    heatmap = np.zeros(activations.shape[:3])  # [B, H, W]

                    for i in range(weights.shape[0]):  # Batch
                        for j in range(weights.shape[1]):  # Channels
                            heatmap[i] += weights[i, j] * activations[i, :, :, j]

                    # Take the first (and only) sample from batch
                    heatmap = heatmap[0]
                else:
                    continue

                # Apply ReLU to focus on positive influences
                heatmap = np.maximum(heatmap, 0)

                # Normalize to [0, 1]
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

                layer_heatmaps.append(heatmap)

        if not layer_heatmaps:
            raise RuntimeError("No valid heatmaps generated. Check target layer names.")

        # Aggregate multiple layer heatmaps
        if len(layer_heatmaps) == 1:
            aggregated_heatmap = layer_heatmaps[0]
        else:
            # Resize all heatmaps to the same shape (largest) before aggregation
            target_shape = max(h.shape for h in layer_heatmaps)
            resized_heatmaps = []
            for hm in layer_heatmaps:
                if hm.shape != target_shape:
                    hm_resized = cv2.resize(hm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                    resized_heatmaps.append(hm_resized)
                else:
                    resized_heatmaps.append(hm)

            if aggregation_method == 'mean':
                aggregated_heatmap = np.mean(resized_heatmaps, axis=0)
            elif aggregation_method == 'max':
                aggregated_heatmap = np.max(resized_heatmaps, axis=0)
            elif aggregation_method == 'sum':
                aggregated_heatmap = np.sum(resized_heatmaps, axis=0)
            elif aggregation_method == 'weighted':
                # Weight later layers more heavily
                weights = np.linspace(0.5, 1.5, len(resized_heatmaps))
                weighted_heatmaps = [h * w for h, w in zip(resized_heatmaps, weights)]
                aggregated_heatmap = np.sum(weighted_heatmaps, axis=0) / np.sum(weights)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            # Re-normalize after aggregation
            if aggregated_heatmap.max() > 0:
                aggregated_heatmap = aggregated_heatmap / aggregated_heatmap.max()

        # Resize to input dimensions if requested
        if resize_to_input and aggregated_heatmap.shape != self.input_shape[:2]:
            heatmap_resized = cv2.resize(
                aggregated_heatmap,
                (self.input_shape[1], self.input_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            return heatmap_resized

        return aggregated_heatmap

    def generate_multi_slice_heatmap(self, input_volume: np.ndarray,
                                   slice_indices: Optional[List[int]] = None,
                                   **kwargs) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for multiple slices of a 3D volume

        Args:
            input_volume: 3D volume [D, H, W, C] or [D, H, W] for grayscale
            slice_indices: List of slice indices to process (if None, process evenly spaced slices)
            **kwargs: Additional arguments for generate_heatmap

        Returns:
            List of heatmaps, one per slice
        """
        if len(input_volume.shape) == 3:
            # Add channel dimension for grayscale
            input_volume = np.expand_dims(input_volume, axis=-1)

        depth = input_volume.shape[0]

        # Select slice indices if not provided
        if slice_indices is None:
            # Select up to 10 evenly spaced slices
            num_slices = min(10, depth)
            slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int).tolist()

        heatmaps = []
        for slice_idx in slice_indices:
            slice_image = input_volume[slice_idx]  # [H, W, C]
            heatmap = self.generate_heatmap(slice_image, **kwargs)
            heatmaps.append(heatmap)

        return heatmaps


def get_default_target_layers(model: tf.keras.Model) -> List[str]:
    """
    Get recommended target layers for Grad-CAM in UNETR model

    Args:
        model: UNETR model instance

    Returns:
        List of recommended layer names
    """
    # Common UNETR layer naming patterns
    decoder_patterns = [
        'conv_block',
        'decoder',
        'upsample',
        'conv_transpose',
        'conv2d',
        'conv2d_transpose'
    ]

    target_layers = []

    # Find decoder convolutional layers
    for layer in reversed(model.layers):  # Start from the end
        layer_name = layer.name.lower()

        # Look for final convolutional layers
        if any(pattern in layer_name for pattern in decoder_patterns):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
                target_layers.append(layer.name)

                # Prioritize layers closer to output (limit to 3-4 layers)
                if len(target_layers) >= 3:
                    break

    if not target_layers:
        # Fallback: find any convolutional layer
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose)):
                target_layers.append(layer.name)
                if len(target_layers) >= 3:
                    break

    return target_layers[:3]  # Return top 3 most relevant layers


def create_gradcam_for_unetr(model_path: str,
                            custom_objects: Optional[Dict] = None,
                            target_layers: Optional[List[str]] = None) -> GradCAM:
    """
    Convenience function to create Grad-CAM instance for UNETR model

    Args:
        model_path: Path to saved model (.h5 file)
        custom_objects: Custom objects for model loading (loss functions, etc.)
        target_layers: Specific target layers (if None, auto-detect)

    Returns:
        GradCAM instance
    """
    # Default custom objects for UNETR
    if custom_objects is None:
        custom_objects = {}

    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Auto-detect target layers if not provided
    if target_layers is None:
        target_layers = get_default_target_layers(model)
        print(f"Auto-detected target layers: {target_layers}")

    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layers, input_shape=(256, 256, 3))

    return gradcam


# Utility functions for testing
def test_gradcam_setup(model_path: str, test_image_path: str):
    """
    Test Grad-CAM setup with a single image

    Args:
        model_path: Path to trained model
        test_image_path: Path to test image
    """
    print("Testing Grad-CAM setup...")

    # Load a sample image
    if os.path.exists(test_image_path):
        image = plt.imread(test_image_path)
        if image.max() > 1.0:
            image = image / 255.0
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
    else:
        # Create dummy image if file doesn't exist
        print(f"Test image not found at {test_image_path}, using dummy image")
        image = np.random.rand(256, 256, 3)

    # Create Grad-CAM
    try:
        gradcam = create_gradcam_for_unetr(model_path)
        heatmap = gradcam.generate_heatmap(image)

        print(f"✓ Grad-CAM setup successful!")
        print(f"  - Input image shape: {image.shape}")
        print(f"  - Heatmap shape: {heatmap.shape}")
        print(f"  - Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"  - Target layers: {gradcam.target_layer_names}")

        return True
    except Exception as e:
        print(f"✗ Grad-CAM setup failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    model_path = "../files/model.h5"
    test_image_path = "../data/test/sample.png"

    if test_gradcam_setup(model_path, test_image_path):
        print("Grad-CAM is ready to use!")
    else:
        print("Please check your model and setup.")