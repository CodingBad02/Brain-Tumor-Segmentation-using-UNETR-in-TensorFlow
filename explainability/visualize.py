"""
Visualization Utilities for Grad-CAM
====================================

This module provides visualization utilities for Grad-CAM heatmaps,
including overlay generation, multi-slice visualization, and batch processing
outputs for brain tumor segmentation explainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import cv2
import os
from typing import List, Tuple, Dict, Optional, Union
import json
from datetime import datetime


def apply_colormap(heatmap: np.ndarray,
                  colormap: str = 'jet',
                  alpha: float = 0.4) -> np.ndarray:
    """
    Apply colormap to heatmap

    Args:
        heatmap: 2D heatmap array [H, W] with values in [0, 1]
        colormap: Matplotlib colormap name
        alpha: Transparency level for overlay

    Returns:
        Colored heatmap as RGB array [H, W, 3]
    """
    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Apply colormap
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Remove alpha channel

    return colored_heatmap


def overlay_heatmap(image: np.ndarray,
                   heatmap: np.ndarray,
                   colormap: str = 'jet',
                   alpha: float = 0.4,
                   resize_method: str = 'bilinear') -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image

    Args:
        image: Original image [H, W] or [H, W, C]
        heatmap: Grad-CAM heatmap [H, W] with values in [0, 1]
        colormap: Colormap for heatmap visualization
        alpha: Transparency for heatmap overlay (0-1)
        resize_method: Method to resize heatmap if needed

    Returns:
        Overlaid image [H, W, 3]
    """
    # Ensure image is 3-channel
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    # Resize heatmap to match image if needed
    if heatmap.shape != image.shape[:2]:
        if resize_method == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif resize_method == 'nearest':
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR

        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=interpolation)

    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0

    # Apply colormap to heatmap
    colored_heatmap = apply_colormap(heatmap, colormap)

    # Blend image and heatmap
    overlaid = (1 - alpha) * image + alpha * colored_heatmap

    return np.clip(overlaid, 0, 1)


def create_comparison_figure(image: np.ndarray,
                           ground_truth: Optional[np.ndarray] = None,
                           prediction: Optional[np.ndarray] = None,
                           heatmap: Optional[np.ndarray] = None,
                           overlay: Optional[np.ndarray] = None,
                           title: str = "Grad-CAM Visualization",
                           figsize: Tuple[int, int] = (20, 5)) -> plt.Figure:
    """
    Create a comparison figure with multiple subplots

    Args:
        image: Original input image
        ground_truth: Ground truth segmentation mask
        prediction: Model prediction
        heatmap: Grad-CAM heatmap
        overlay: Overlay of heatmap on image
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Determine number of subplots
    subplots = []
    subplot_titles = []

    # Always include original image
    subplots.append(image)
    subplot_titles.append("Input Image")

    # Add optional subplots
    if ground_truth is not None:
        subplots.append(ground_truth)
        subplot_titles.append("Ground Truth")

    if prediction is not None:
        subplots.append(prediction)
        subplot_titles.append("Prediction")

    if heatmap is not None:
        subplots.append(heatmap)
        subplot_titles.append("Grad-CAM Heatmap")

    if overlay is not None:
        subplots.append(overlay)
        subplot_titles.append("Grad-CAM Overlay")

    # Create figure
    n_cols = len(subplots)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # Plot each subplot
    for idx, (ax, img, title) in enumerate(zip(axes, subplots, subplot_titles)):
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        im = ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def visualize_multi_slice(volume: np.ndarray,
                         heatmaps: List[np.ndarray],
                         slice_indices: List[int],
                         predictions: Optional[List[np.ndarray]] = None,
                         ground_truths: Optional[List[np.ndarray]] = None,
                         save_path: Optional[str] = None,
                         show: bool = False) -> plt.Figure:
    """
    Visualize multiple slices with their corresponding heatmaps

    Args:
        volume: 3D volume [D, H, W] or [D, H, W, C]
        heatmaps: List of heatmaps for each slice
        slice_indices: List of slice indices being visualized
        predictions: Optional list of predictions for each slice
        ground_truths: Optional list of ground truth masks for each slice
        save_path: Optional path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))

    if n_slices == 1:
        axes = axes.reshape(1, -1)

    for i, slice_idx in enumerate(slice_indices):
        # Get slice
        if len(volume.shape) == 4:
            slice_img = volume[slice_idx]
        else:
            slice_img = volume[slice_idx]

        # Ensure 3-channel for display
        if len(slice_img.shape) == 2:
            slice_img = np.stack([slice_img] * 3, axis=-1)

        # Display original image
        axes[i, 0].imshow(slice_img)
        axes[i, 0].set_title(f"Slice {slice_idx} - Original", fontweight='bold')
        axes[i, 0].axis('off')

        # Display ground truth if available
        if ground_truths and i < len(ground_truths):
            gt = ground_truths[i]
            axes[i, 1].imshow(gt, cmap='gray')
            axes[i, 1].set_title("Ground Truth", fontweight='bold')
        else:
            axes[i, 1].text(0.5, 0.5, 'No Ground Truth',
                           ha='center', va='center', transform=axes[i, 1].transAxes)
        axes[i, 1].axis('off')

        # Display prediction if available
        if predictions and i < len(predictions):
            pred = predictions[i]
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title("Prediction", fontweight='bold')
        else:
            axes[i, 2].text(0.5, 0.5, 'No Prediction',
                           ha='center', va='center', transform=axes[i, 2].transAxes)
        axes[i, 2].axis('off')

        # Display heatmap overlay
        if i < len(heatmaps):
            overlay = overlay_heatmap(slice_img, heatmaps[i])
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("Grad-CAM Overlay", fontweight='bold')
        else:
            axes[i, 3].text(0.5, 0.5, 'No Heatmap',
                           ha='center', va='center', transform=axes[i, 3].transAxes)
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-slice visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def save_gradcam_results(image: np.ndarray,
                        heatmap: np.ndarray,
                        overlay: np.ndarray,
                        prediction: Optional[np.ndarray] = None,
                        ground_truth: Optional[np.ndarray] = None,
                        save_dir: str = "results/gradcam",
                        case_id: str = "case_001",
                        metadata: Optional[Dict] = None) -> Dict[str, str]:
    """
    Save Grad-CAM results and metadata

    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        overlay: Heatmap overlay on image
        prediction: Optional prediction mask
        ground_truth: Optional ground truth mask
        save_dir: Directory to save results
        case_id: Unique identifier for the case
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with saved file paths
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Prepare metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        'case_id': case_id,
        'timestamp': datetime.now().isoformat(),
        'image_shape': image.shape,
        'heatmap_shape': heatmap.shape,
        'heatmap_stats': {
            'min': float(heatmap.min()),
            'max': float(heatmap.max()),
            'mean': float(heatmap.mean()),
            'std': float(heatmap.std())
        }
    })

    saved_files = {}

    # Save original image
    image_path = os.path.join(save_dir, f"{case_id}_original.png")
    plt.imsave(image_path, image)
    saved_files['original'] = image_path

    # Save heatmap
    heatmap_path = os.path.join(save_dir, f"{case_id}_heatmap.png")
    plt.imsave(heatmap_path, heatmap, cmap='jet')
    saved_files['heatmap'] = heatmap_path

    # Save overlay
    overlay_path = os.path.join(save_dir, f"{case_id}_overlay.png")
    plt.imsave(overlay_path, overlay)
    saved_files['overlay'] = overlay_path

    # Save prediction if available
    if prediction is not None:
        pred_path = os.path.join(save_dir, f"{case_id}_prediction.png")
        plt.imsave(pred_path, prediction, cmap='gray')
        saved_files['prediction'] = pred_path

    # Save ground truth if available
    if ground_truth is not None:
        gt_path = os.path.join(save_dir, f"{case_id}_ground_truth.png")
        plt.imsave(gt_path, ground_truth, cmap='gray')
        saved_files['ground_truth'] = gt_path

    # Create and save comparison figure
    fig = create_comparison_figure(
        image=image,
        ground_truth=ground_truth,
        prediction=prediction,
        heatmap=heatmap,
        overlay=overlay,
        title=f"Grad-CAM Analysis - {case_id}"
    )

    comparison_path = os.path.join(save_dir, f"{case_id}_comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files['comparison'] = comparison_path

    # Save metadata
    metadata_path = os.path.join(save_dir, f"{case_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files['metadata'] = metadata_path

    return saved_files


def create_tensorboard_images(images: List[np.ndarray],
                             heatmaps: List[np.ndarray],
                             overlays: List[np.ndarray],
                             case_ids: List[str],
                             step: int = 0,
                             log_dir: str = "logs/gradcam") -> None:
    """
    Create TensorBoard image summaries for Grad-CAM results

    Args:
        images: List of original images
        heatmaps: List of heatmaps
        overlays: List of overlays
        case_ids: List of case identifiers
        step: Training/evaluation step
        log_dir: TensorBoard log directory
    """
    import tensorflow as tf
    from datetime import datetime

    # Create log writer
    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        # Log original images
        if images:
            images_tensor = tf.convert_to_tensor(np.stack(images))
            tf.summary.image("Original Images", images_tensor, step=step, max_outputs=len(images))

        # Log heatmaps
        if heatmaps:
            # Convert heatmaps to 3-channel for TensorBoard
            heatmap_colors = [apply_colormap(h) for h in heatmaps]
            heatmaps_tensor = tf.convert_to_tensor(np.stack(heatmap_colors))
            tf.summary.image("Grad-CAM Heatmaps", heatmaps_tensor, step=step, max_outputs=len(heatmaps))

        # Log overlays
        if overlays:
            overlays_tensor = tf.convert_to_tensor(np.stack(overlays))
            tf.summary.image("Grad-CAM Overlays", overlays_tensor, step=step, max_outputs=len(overlays))

        # Log case IDs as text
        if case_ids:
            case_text = "\n".join([f"Case {i}: {case_id}" for i, case_id in enumerate(case_ids)])
            tf.summary.text("Case IDs", case_text, step=step)

    writer.flush()
    print(f"Logged {len(images)} Grad-CAM results to TensorBoard at step {step}")


def generate_report(results_dir: str = "results/gradcam",
                   output_path: str = "results/gradcam_report.html") -> str:
    """
    Generate HTML report summarizing Grad-CAM results

    Args:
        results_dir: Directory containing Grad-CAM results
        output_path: Path to save HTML report

    Returns:
        Path to generated report
    """
    # Find all result files
    case_files = []
    for file in os.listdir(results_dir):
        if file.endswith("_metadata.json"):
            case_id = file.replace("_metadata.json", "")
            case_files.append(case_id)

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grad-CAM Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .case {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; }}
            .case h2 {{ color: #0066cc; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; height: auto; }}
            .metadata {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Grad-CAM Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total cases analyzed: {len(case_files)}</p>
    """

    # Add each case to report
    for case_id in sorted(case_files):
        metadata_path = os.path.join(results_dir, f"{case_id}_metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        html_content += f"""
        <div class="case">
            <h2>{case_id}</h2>
            <div class="metadata">
                <p><strong>Timestamp:</strong> {metadata.get('timestamp', 'N/A')}</p>
                <p><strong>Image Shape:</strong> {metadata.get('image_shape', 'N/A')}</p>
                <p><strong>Heatmap Stats:</strong> Min={metadata.get('heatmap_stats', {}).get('min', 'N/A'):.3f},
                   Max={metadata.get('heatmap_stats', {}).get('max', 'N/A'):.3f},
                   Mean={metadata.get('heatmap_stats', {}).get('mean', 'N/A'):.3f}</p>
            </div>
            <div class="image-grid">
        """

        # Add images
        for image_type in ['original', 'ground_truth', 'prediction', 'heatmap', 'overlay', 'comparison']:
            image_path = os.path.join(results_dir, f"{case_id}_{image_type}.png")
            if os.path.exists(image_path):
                display_name = image_type.replace('_', ' ').title()
                html_content += f"""
                <div class="image-item">
                    <h3>{display_name}</h3>
                    <img src="{os.path.relpath(image_path, os.path.dirname(output_path))}" alt="{display_name}">
                </div>
                """

        html_content += "</div></div>"

    html_content += """
    </body>
    </html>
    """

    # Save HTML report
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Generated Grad-CAM report: {output_path}")
    return output_path


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    image = np.random.rand(256, 256, 3)
    heatmap = np.random.rand(256, 256)
    overlay = overlay_heatmap(image, heatmap, alpha=0.5)

    # Save test results
    saved_files = save_gradcam_results(
        image=image,
        heatmap=heatmap,
        overlay=overlay,
        case_id="test_001",
        metadata={"test": True, "model_version": "UNETR-v2"}
    )

    print("Test visualization completed!")
    print("Saved files:", saved_files)