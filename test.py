import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from patchify import patchify
from train import load_dataset, create_dir
from metrics import dice_loss, dice_coef
import argparse
import sys

# Add explainability module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'explainability'))
try:
    from gradcam import GradCAM, get_default_target_layers
    from visualize import save_gradcam_results, overlay_heatmap
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: Grad-CAM modules not available. Install required dependencies for explainability features.")


""" UNETR  Configration """
cf = {}
cf["image_size"] = 256
cf["num_channels"] = 3
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["num_heads"] = 6
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"]*cf["patch_size"]*cf["num_channels"]
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test UNETR model with optional Grad-CAM explainability')

    # Test configuration
    parser.add_argument('--dataset', type=str, default='BraTS_data',
                       help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='files/model.h5',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')

    # Grad-CAM options
    parser.add_argument('--gradcam', action='store_true',
                       help='Enable Grad-CAM heatmap generation')
    parser.add_argument('--gradcam-layer', type=str, nargs='+',
                       help='Target layer(s) for Grad-CAM (auto-detected if not specified)')
    parser.add_argument('--gradcam-output', type=str, default='results/gradcam',
                       help='Output directory for Grad-CAM results')
    parser.add_argument('--gradcam-samples', type=int, default=10,
                       help='Number of test samples to generate Grad-CAM for (0 = all)')
    parser.add_argument('--gradcam-alpha', type=float, default=0.4,
                       help='Transparency for Grad-CAM overlay (0-1)')
    parser.add_argument('--gradcam-colormap', type=str, default='jet',
                       help='Colormap for Grad-CAM visualization')

    # Sampling options
    parser.add_argument('--test-samples', type=int, default=0,
                       help='Number of test samples to process (0 = all)')

    return parser.parse_args()


def process_sample_with_gradcam(model, image_path, mask_path, name, gradcam, args):
    """Process a single sample with Grad-CAM visualization"""
    """ Extracting the name """
    name = name.split("/")[-1].split('.')[0]  # Remove extension for cleaner naming

    """ Reading the image """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    x = image / 255.0

    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(x, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    """ Read Mask """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    """ Prediction """
    pred = model.predict(patches, verbose=0)[0]
    pred = np.concatenate([pred, pred, pred], axis=-1)

    """ Generate Grad-CAM """
    try:
        heatmap = gradcam.generate_heatmap(x, aggregation_method='weighted')
        overlay = overlay_heatmap(x, heatmap, alpha=args.gradcam_alpha, colormap=args.gradcam_colormap)

        """ Save Grad-CAM results """
        metadata = {
            'model_path': args.model,
            'target_layers': gradcam.target_layer_names,
            'gradcam_alpha': args.gradcam_alpha,
            'gradcam_colormap': args.gradcam_colormap,
            'original_image_path': image_path
        }

        saved_files = save_gradcam_results(
            image=x,
            heatmap=heatmap,
            overlay=overlay,
            prediction=pred[:, :, 0],  # Single channel
            ground_truth=mask[:, :, 0],  # Single channel
            save_dir=args.gradcam_output,
            case_id=name,
            metadata=metadata
        )

        print(f"  ✓ Grad-CAM saved for {name}")

    except Exception as e:
        print(f"  ✗ Grad-CAM failed for {name}: {e}")
        overlay = None

    """ Save standard comparison image """
    line = np.ones((cf["image_size"], 10, 3)) * 255
    if overlay is not None:
        # Include overlay in comparison
        cat_images = np.concatenate([image, line, mask*255, line, pred*255, line, overlay*255], axis=1)
    else:
        # Standard comparison
        cat_images = np.concatenate([image, line, mask*255, line, pred*255], axis=1)

    save_image_path = os.path.join(args.output, f"{name}_comparison.png")
    cv2.imwrite(save_image_path, cat_images)

    return True


if __name__ == "__main__":
    """ Parse arguments """
    args = parse_args()

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir(args.output)
    if args.gradcam:
        create_dir(args.gradcam_output)

    """ Load the model """
    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})

    """ Initialize Grad-CAM if requested """
    gradcam = None
    if args.gradcam and GRADCAM_AVAILABLE:
        print("Initializing Grad-CAM...")
        try:
            # Determine target layers
            if args.gradcam_layer:
                target_layers = args.gradcam_layer
                print(f"Using specified target layers: {target_layers}")
            else:
                target_layers = get_default_target_layers(model)
                print(f"Auto-detected target layers: {target_layers}")

            # Create Grad-CAM instance
            gradcam = GradCAM(model, target_layers, input_shape=(cf["image_size"], cf["image_size"], cf["num_channels"]))
            print(f"✓ Grad-CAM initialized with {len(target_layers)} target layer(s)")

        except Exception as e:
            print(f"✗ Failed to initialize Grad-CAM: {e}")
            print("Continuing without Grad-CAM...")
            args.gradcam = False

    """ Dataset """
    print(f"Loading dataset from: {args.dataset}")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(args.dataset)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Determine number of samples to process """
    if args.test_samples > 0:
        test_x = test_x[:args.test_samples]
        test_y = test_y[:args.test_samples]
        print(f"Processing {len(test_x)} test samples (limited by --test-samples)")

    if args.gradcam and args.gradcam_samples > 0:
        gradcam_samples = min(args.gradcam_samples, len(test_x))
        print(f"Generating Grad-CAM for {gradcam_samples} samples (limited by --gradcam-samples)")
    else:
        gradcam_samples = len(test_x) if args.gradcam else 0

    """ Prediction """
    print("\nStarting prediction...")

    if args.gradcam and gradcam_samples > 0:
        # Process with Grad-CAM for first N samples
        print(f"\nGenerating Grad-CAM for first {gradcam_samples} samples...")
        gradcam_progress = tqdm(zip(test_x[:gradcam_samples], test_y[:gradcam_samples]),
                               total=gradcam_samples, desc="Grad-CAM")

        for x, y in gradcam_progress:
            process_sample_with_gradcam(model, x, y, x, gradcam, args)

        # Process remaining samples without Grad-CAM
        if gradcam_samples < len(test_x):
            print(f"\nProcessing remaining {len(test_x) - gradcam_samples} samples without Grad-CAM...")
            remaining_progress = tqdm(zip(test_x[gradcam_samples:], test_y[gradcam_samples:]),
                                   total=len(test_x) - gradcam_samples, desc="Prediction")

            for x, y in remaining_progress:
                """ Extracting the name """
                name = x.split("/")[-1]

                """ Reading the image """
                image = cv2.imread(x, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
                x_img = image / 255.0

                patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
                patches = patchify(x_img, patch_shape, cf["patch_size"])
                patches = np.reshape(patches, cf["flat_patches_shape"])
                patches = patches.astype(np.float32)
                patches = np.expand_dims(patches, axis=0)

                """ Read Mask """
                mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
                mask = mask / 255.0
                mask = np.expand_dims(mask, axis=-1)
                mask = np.concatenate([mask, mask, mask], axis=-1)

                """ Prediction """
                pred = model.predict(patches, verbose=0)[0]
                pred = np.concatenate([pred, pred, pred], axis=-1)

                """ Save final mask """
                line = np.ones((cf["image_size"], 10, 3)) * 255
                cat_images = np.concatenate([image, line, mask*255, line, pred*255], axis=1)
                save_image_path = os.path.join(args.output, name)
                cv2.imwrite(save_image_path, cat_images)
    else:
        # Standard processing without Grad-CAM
        progress_bar = tqdm(zip(test_x, test_y), total=len(test_x), desc="Prediction")

        for x, y in progress_bar:
            """ Extracting the name """
            name = x.split("/")[-1]

            """ Reading the image """
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
            x_img = image / 255.0

            patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
            patches = patchify(x_img, patch_shape, cf["patch_size"])
            patches = np.reshape(patches, cf["flat_patches_shape"])
            patches = patches.astype(np.float32)
            patches = np.expand_dims(patches, axis=0)

            """ Read Mask """
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)
            mask = np.concatenate([mask, mask, mask], axis=-1)

            """ Prediction """
            pred = model.predict(patches, verbose=0)[0]
            pred = np.concatenate([pred, pred, pred], axis=-1)

            """ Save final mask """
            line = np.ones((cf["image_size"], 10, 3)) * 255
            cat_images = np.concatenate([image, line, mask*255, line, pred*255], axis=1)
            save_image_path = os.path.join(args.output, name)
            cv2.imwrite(save_image_path, cat_images)

    print(f"\n✓ Processing complete! Results saved to: {args.output}")
    if args.gradcam and gradcam_samples > 0:
        print(f"✓ Grad-CAM results saved to: {args.gradcam_output}")
        print(f"  Processed {gradcam_samples} samples with Grad-CAM")

    # Generate HTML report if Grad-CAM was used
    if args.gradcam and GRADCAM_AVAILABLE:
        try:
            from visualize import generate_report
            report_path = os.path.join(args.gradcam_output, "gradcam_report.html")
            generate_report(args.gradcam_output, report_path)
            print(f"✓ HTML report generated: {report_path}")
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")










    ## ...
