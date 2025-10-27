# Explainability Module: Grad-CAM for UNETR
=============================================

This directory contains the Grad-CAM (Gradient-weighted Class Activation Mapping) implementation specifically designed for the UNETR brain tumor segmentation model.

## Overview

Grad-CAM provides visual explanations for model predictions by highlighting the regions of the input image that were most important for the segmentation decision. This helps understand:

- **Where** the model is "looking" when making predictions
- **What features** contribute to tumor detection
- **Model behavior** on specific cases
- **Potential biases** or failure modes

## Files

- **`gradcam.py`**: Core Grad-CAM implementation with the `GradCAM` class
- **`visualize.py`**: Visualization utilities for heatmaps, overlays, and reports
- **`README.md`**: This file

## Quick Start

### Basic Usage

```python
from explainability.gradcam import GradCAM, create_gradcam_for_unetr
from explainability.visualize import overlay_heatmap, save_gradcam_results

# Load model and create Grad-CAM
gradcam = create_gradcam_for_unetr("files/model.h5")

# Generate heatmap for an image
image = cv2.imread("test_image.png")
heatmap = gradcam.generate_heatmap(image)

# Create overlay
overlay = overlay_heatmap(image, heatmap, alpha=0.4)

# Save results
save_gradcam_results(image, heatmap, overlay, case_id="test_case")
```

### Command Line Usage

```bash
# Generate Grad-CAM for 10 test samples
python test.py --gradcam --gradcam-samples 10

# Process all test samples with Grad-CAM
python test.py --gradcam --gradcam-samples 0

# Use specific target layers
python test.py --gradcam --gradcam-layer conv_block_7 conv_block_6
```

## Features

### 1. Automatic Layer Detection
The implementation automatically identifies optimal convolutional layers in the UNETR model:

- **conv_block_7**: Final decoder layer (highest spatial resolution)
- **conv_block_6**: Mid-level decoder features
- **conv_block_5**: Higher-level semantic features

### 2. Multi-layer Aggregation
Combine features from multiple layers using different strategies:

```python
# Weighted aggregation (default, gives more weight to later layers)
heatmap = gradcam.generate_heatmap(image, aggregation_method='weighted')

# Other options
heatmap = gradcam.generate_heatmap(image, aggregation_method='mean')  # Average
heatmap = gradcam.generate_heatmap(image, aggregation_method='max')   # Maximum
heatmap = gradcam.generate_heatmap(image, aggregation_method='sum')   # Sum
```

### 3. Flexible Visualization
Customize the appearance of heatmaps:

```python
# Different colormaps
overlay = overlay_heatmap(image, heatmap, colormap='hot')  # or 'jet', 'cool', 'viridis'

# Adjust transparency
overlay = overlay_heatmap(image, heatmap, alpha=0.5)  # 0.0 to 1.0
```

### 4. Comprehensive Output
For each case, the following files are generated:

- `*_original.png`: Original input image
- `*_heatmap.png`: Grad-CAM heatmap
- `*_overlay.png`: Heatmap overlay on image
- `*_prediction.png`: Model prediction mask
- `*_ground_truth.png`: Ground truth mask (if available)
- `*_comparison.png`: Side-by-side comparison
- `*_metadata.json`: Processing metadata and metrics

## API Reference

### GradCAM Class

```python
class GradCAM:
    def __init__(self, model, target_layer_names, input_shape=(256, 256, 3)):
        """
        Initialize Grad-CAM for UNETR model

        Args:
            model: Loaded UNETR model
            target_layer_names: List of layer names for Grad-CAM
            input_shape: Input image shape
        """

    def generate_heatmap(self, input_image, target_class_idx=0, aggregation_method='mean'):
        """
        Generate Grad-CAM heatmap

        Args:
            input_image: Input image [H, W, C] or [1, H, W, C]
            target_class_idx: Target class index (for binary, use 0)
            aggregation_method: How to aggregate multiple layers

        Returns:
            Heatmap array [H, W] with values in [0, 1]
        """
```

### Utility Functions

```python
def get_default_target_layers(model):
    """Auto-detect optimal target layers for Grad-CAM"""

def create_gradcam_for_unetr(model_path, custom_objects=None, target_layers=None):
    """Convenience function to create Grad-CAM for UNETR model"""

def overlay_heatmap(image, heatmap, colormap='jet', alpha=0.4):
    """Create overlay of heatmap on original image"""

def save_gradcam_results(image, heatmap, overlay, ...):
    """Save all Grad-CAM results and metadata"""
```

## Best Practices

### 1. Layer Selection
- Use **conv_block_7** for fine-grained localization
- Use **conv_block_5** for more semantic/feature-level explanations
- Use multiple layers for comprehensive understanding

### 2. Interpretation
- **Bright regions** in heatmap indicate high importance
- **Overlay alignment** with tumor regions validates model focus
- **Out-of-focus heatmaps** may indicate model confusion

### 3. Quality Checks
- Ensure heatmaps align with anatomical features
- Verify consistent behavior across similar cases
- Check for systematic biases or attention patterns

## Troubleshooting

### Common Issues

1. **"No target layers found"**
   - Check model architecture and layer names
   - Use `get_default_target_layers()` to see available options

2. **Blank or uniform heatmaps**
   - Model may not be well-trained
   - Try different target layers
   - Check gradient flow with `tf.debugging.check_numerics`

3. **Memory errors**
   - Reduce batch size
   - Use fewer target layers
   - Process images sequentially

4. **Slow performance**
   - Use fewer layers for Grad-CAM
   - Enable parallel processing for batch jobs
   - Consider using GPU for gradient computation

### Debug Tips

```python
# Check available layers
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)

# Verify gradients are computed
with tf.GradientTape() as tape:
    predictions = model(input_image)
    loss = tf.reduce_mean(predictions)
gradients = tape.gradient(loss, target_layer_output)
print(f"Gradient norm: {tf.norm(gradients)}")
```

## Integration with Training

The Grad-CAM implementation can be integrated with training workflows:

```python
# During validation
if epoch % 10 == 0:  # Every 10 epochs
    sample_images = get_validation_samples()
    gradcam = create_gradcam_for_unetr(model_path)
    for img in sample_images:
        heatmap = gradcam.generate_heatmap(img)
        # Log to TensorBoard
        tf.summary.image("gradcam", overlay, step=epoch)
```

## Advanced Features

### Custom Layer Weights
Specify custom weights for different layers:

```python
# Example: Give more weight to final layer
layer_weights = {
    'conv_block_5': 0.3,
    'conv_block_6': 0.3,
    'conv_block_7': 0.4
}
```

### Multi-class Segmentation
For multi-class tumor segmentation (future enhancement):

```python
# Generate heatmap for specific class
heatmap = gradcam.generate_heatmap(image, target_class_idx=2)  # Class 2
```

### Temporal Analysis
Track how attention changes during training:

```python
# Save heatmaps at different epochs
for epoch in range(0, 100, 10):
    model = load_model(f"checkpoints/model_epoch_{epoch}.h5")
    gradcam = GradCAM(model, target_layers)
    heatmap = gradcam.generate_heatmap(sample_image)
    save_heatmap(heatmap, f"epoch_{epoch}.png")
```

## References

1. Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
2. UNETR: Hatamizadeh, V., et al. "UNETR: Transformers for 3D Medical Image Segmentation." WACV 2022.

## Contributing

To extend the explainability module:

1. Add new visualization methods to `visualize.py`
2. Implement additional attribution methods (e.g., Integrated Gradients)
3. Create custom aggregation strategies for multi-layer features
4. Add support for 3D volumes and multi-modal inputs

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the main repository README
3. Open an issue with:
   - Model architecture details
   - Error messages and stack traces
   - Sample input/output if possible