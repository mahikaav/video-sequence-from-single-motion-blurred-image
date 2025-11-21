"""
Script to reproduce results from "Learning to Extract a Video Sequence from a Single Motion-Blurred Image"
Jin et al., CVPR 2018

This script:
1. Loads GoPro test sequences with pre-existing blur images
2. Downsamples to 45% resolution
3. Runs the model inference using their actual model architecture
4. Calculates PSNR for middle frame prediction

Usage:
    python evaluate_gopro.py --gopro_path /path/to/GOPRO_Large/test --cuda
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from tqdm import tqdm

# Import their models (make sure model.py is in the same directory)
from model import centerEsti, F26_N9, F17_N9, F35_N8


def load_gopro_sequence(sequence_path):
    """Load blurry and sharp frames from a GoPro sequence."""
    blur_dir = os.path.join(sequence_path, 'blur')
    sharp_dir = os.path.join(sequence_path, 'sharp')
    
    if not os.path.exists(blur_dir) or not os.path.exists(sharp_dir):
        print(f"Warning: blur or sharp folder not found in {sequence_path}")
        return [], []
    
    # Load pre-existing blur images
    blur_files = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
    sharp_files = sorted(glob.glob(os.path.join(sharp_dir, '*.png')))
    
    blurs = []
    sharps = []
    
    for bf in blur_files:
        img = Image.open(bf).convert('RGB')
        blurs.append(np.array(img))
    
    for sf in sharp_files:
        img = Image.open(sf).convert('RGB')
        sharps.append(np.array(img))
    
    return blurs, sharps


def create_blur_samples(blurs, sharps, downsample_factor=0.45):
    """
    Create test samples from pre-existing blur images.
    Each blur corresponds to its matching sharp frame.
    
    Args:
        blurs: List of blurry images (numpy arrays)
        sharps: List of sharp images (numpy arrays)
        downsample_factor: Downsampling factor (paper uses 0.45 = 45%)
    
    Returns:
        List of dictionaries containing blurry images and ground truth frames
    """
    samples = []
    
    # Assume each blur image corresponds to its matching sharp frame
    for blur, sharp in zip(blurs, sharps):
        # Downsample to 45%
        h, w = blur.shape[:2]
        new_h, new_w = int(h * downsample_factor), int(w * downsample_factor)
        
        blur_pil = Image.fromarray(blur)
        blur_resized = np.array(blur_pil.resize((new_w, new_h), Image.BILINEAR))
        
        sharp_pil = Image.fromarray(sharp)
        sharp_resized = np.array(sharp_pil.resize((new_w, new_h), Image.BILINEAR))
        
        samples.append({
            'blurry': blur_resized,
            'middle_frame': sharp_resized,
        })
    
    return samples


def prepare_input(blurry_img):
    """
    Prepare input tensor for the model (same as demo.py).
    
    Args:
        blurry_img: PIL Image or numpy array
    
    Returns:
        Tensor ready for model input
    """
    if isinstance(blurry_img, np.ndarray):
        blurry_img = Image.fromarray(blurry_img)
    
    # Crop to multiple of 20 (as done in demo.py)
    width, height = blurry_img.size
    blurry_img = blurry_img.crop((0, 0, width - width % 20, height - height % 20))
    
    # Convert to tensor
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = input_transform(blurry_img)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return input_tensor


def remove_running_stats(state_dict):
    """Remove running_mean and running_var from InstanceNorm2d layers"""
    keys_to_remove = []
    for key in state_dict.keys():
        if 'running_mean' in key or 'running_var' in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict


def load_models(cuda=True):
    """
    Load all 4 pretrained models (same as demo.py).
    
    Returns:
        Tuple of (model1, model2, model3, model4)
    """
    print("Loading models...")
    
    model1 = centerEsti()  # Middle frame (frame 4)
    model2 = F35_N8()      # Frames 3 and 5
    model3 = F26_N9()      # Frames 2 and 6
    model4 = F17_N9()      # Frames 1 and 7
    
    map_location = None if cuda else torch.device('cpu')
    
    # Load pretrained weights
    checkpoint = torch.load('models/center_v3.pth', map_location=map_location, weights_only=False)
    model1.load_state_dict(remove_running_stats(checkpoint['state_dict_G']), strict=False)
    
    checkpoint = torch.load('models/F35_N8.pth', map_location=map_location, weights_only=False)
    model2.load_state_dict(remove_running_stats(checkpoint['state_dict_G']), strict=False)
    
    checkpoint = torch.load('models/F26_N9_from_F35_N8.pth', map_location=map_location, weights_only=False)
    model3.load_state_dict(remove_running_stats(checkpoint['state_dict_G']), strict=False)
    
    checkpoint = torch.load('models/F17_N9_from_F26_N9_from_F35_N8.pth', map_location=map_location, weights_only=False)
    model4.load_state_dict(remove_running_stats(checkpoint['state_dict_G']), strict=False)
    
    if cuda:
        model1.cuda()
        model2.cuda()
        model3.cuda()
        model4.cuda()
    
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    
    print("Models loaded successfully!")
    return model1, model2, model3, model4


def predict_frames(models, blurry_input, cuda=True):
    """
    Run inference to predict all 7 frames.
    
    Args:
        models: Tuple of (model1, model2, model3, model4)
        blurry_input: Input tensor
        cuda: Whether to use CUDA
    
    Returns:
        List of 7 predicted frames as numpy arrays
    """
    model1, model2, model3, model4 = models
    
    if cuda:
        blurry_input = blurry_input.cuda()
    
    with torch.no_grad():
        # Sequential prediction (same as demo.py)
        output4 = model1(blurry_input)  # Middle frame
        output3_5 = model2(blurry_input, output4)  # Frames 3, 5
        output2_6 = model3(blurry_input, output3_5[0], output4, output3_5[1])  # Frames 2, 6
        output1_7 = model4(blurry_input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])  # Frames 1, 7
    
    # Move to CPU and convert to numpy
    if cuda:
        output1 = output1_7[0].cpu()
        output2 = output2_6[0].cpu()
        output3 = output3_5[0].cpu()
        output4 = output4.cpu()
        output5 = output3_5[1].cpu()
        output6 = output2_6[1].cpu()
        output7 = output1_7[1].cpu()
    else:
        output1 = output1_7[0]
        output2 = output2_6[0]
        output3 = output3_5[0]
        output4 = output4
        output5 = output3_5[1]
        output6 = output2_6[1]
        output7 = output1_7[1]
    
    # Convert to numpy arrays (H, W, C format) and scale to [0, 255]
    predicted_frames = []
    for output in [output1, output2, output3, output4, output5, output6, output7]:
        frame = output.data[0].numpy()  # Remove batch dim
        frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        predicted_frames.append(frame)
    
    return predicted_frames


def evaluate_middle_frame(models, samples, cuda=True):
    """
    Evaluate middle frame prediction on all samples.
    
    Args:
        models: Tuple of loaded models
        samples: List of test samples
        cuda: Whether to use CUDA
    
    Returns:
        Dictionary with PSNR scores and predictions
    """
    psnr_scores = []
    predictions = []
    
    print(f"\nEvaluating on {len(samples)} samples...")
    
    for idx, sample in enumerate(tqdm(samples)):
        # Prepare input
        blurry_input = prepare_input(sample['blurry'])
        
        # Predict all 7 frames
        pred_frames = predict_frames(models, blurry_input, cuda)
        
        # We care about the middle frame (frame 4, index 3)
        pred_middle = pred_frames[3]
        gt_middle = sample['middle_frame']
        
        # Handle size mismatch due to crop to multiple of 20
        # Crop ground truth to match prediction size
        h_pred, w_pred = pred_middle.shape[:2]
        h_gt, w_gt = gt_middle.shape[:2]
        if h_pred != h_gt or w_pred != w_gt:
            gt_middle = gt_middle[:h_pred, :w_pred, :]
        
        # Calculate PSNR
        score = psnr(gt_middle, pred_middle, data_range=255)
        psnr_scores.append(score)
        
        # Debug first sample
        if idx == 0:
            print(f"\n{'='*60}")
            print("DEBUG FIRST SAMPLE:")
            print(f"{'='*60}")
            print(f"Blurry input shape: {sample['blurry'].shape}")
            print(f"Blurry input range: [{sample['blurry'].min()}, {sample['blurry'].max()}]")
            print(f"\nGT middle shape: {gt_middle.shape}")
            print(f"GT middle range: [{gt_middle.min()}, {gt_middle.max()}]")
            print(f"\nPred middle shape: {pred_middle.shape}")
            print(f"Pred middle range: [{pred_middle.min()}, {pred_middle.max()}]")
            print(f"\nInput tensor shape: {blurry_input.shape}")
            print(f"Input tensor range: [{blurry_input.min():.4f}, {blurry_input.max():.4f}]")
            print(f"\nPSNR: {score:.2f} dB")
            
            # Check if model is just copying input
            input_pred_diff = np.abs(sample['blurry'][:h_pred, :w_pred, :].astype(float) - pred_middle.astype(float)).mean()
            input_gt_diff = np.abs(sample['blurry'][:h_pred, :w_pred, :].astype(float) - gt_middle.astype(float)).mean()
            print(f"\nMean abs difference (Input vs Prediction): {input_pred_diff:.2f}")
            print(f"Mean abs difference (Input vs GT): {input_gt_diff:.2f}")
            print(f"{'='*60}\n")
        
        predictions.append({
            'predicted_middle': pred_middle,
            'predicted_all': pred_frames,
            'ground_truth_middle': gt_middle,
            'blurry_input': sample['blurry'],
            'psnr': score
        })
        
        if (idx + 1) % 100 == 0:
            print(f"\nProcessed {idx + 1}/{len(samples)} samples")
            print(f"Average PSNR so far: {np.mean(psnr_scores):.2f} dB")
    
    return {
        'mean_psnr': np.mean(psnr_scores),
        'std_psnr': np.std(psnr_scores),
        'all_psnr': psnr_scores,
        'predictions': predictions
    }


def save_results(results, output_dir='evaluation_results'):
    """Save evaluation results and sample predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PSNR statistics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Number of test samples: {len(results['all_psnr'])}")
    print(f"Mean PSNR: {results['mean_psnr']:.4f} dB")
    print(f"Std PSNR: {results['std_psnr']:.4f} dB")
    print(f"{'='*60}")
    print(f"\nPaper reports:")
    print(f"  32.20 dB on their test set (45% downsampled)")
    print(f"  29.02 dB on their custom sequences")
    print(f"  28.19 dB on Nah's test set (45% downsampled)")
    print(f"{'='*60}")
    
    with open(os.path.join(output_dir, 'psnr_results.txt'), 'w') as f:
        f.write(f"Number of samples: {len(results['all_psnr'])}\n")
        f.write(f"Mean PSNR: {results['mean_psnr']:.4f} dB\n")
        f.write(f"Std PSNR: {results['std_psnr']:.4f} dB\n")
        f.write(f"\nAll PSNR scores:\n")
        for i, score in enumerate(results['all_psnr']):
            f.write(f"Sample {i+1}: {score:.4f} dB\n")
    
    # Save first 10 predictions as images
    print(f"\nSaving sample predictions to {output_dir}/...")
    for idx, pred in enumerate(results['predictions'][:10]):
        sample_dir = os.path.join(output_dir, f'sample_{idx+1:04d}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save blurry input
        Image.fromarray(pred['blurry_input']).save(
            os.path.join(sample_dir, '0_input_blurry.png'))
        
        # Save middle frame comparison
        Image.fromarray(pred['predicted_middle']).save(
            os.path.join(sample_dir, '4_predicted_middle.png'))
        Image.fromarray(pred['ground_truth_middle']).save(
            os.path.join(sample_dir, '4_ground_truth_middle.png'))
        
        # Save all 7 predicted frames
        for frame_idx, frame in enumerate(pred['predicted_all'], 1):
            Image.fromarray(frame).save(
                os.path.join(sample_dir, f'{frame_idx}_predicted_frame.png'))
        
        with open(os.path.join(sample_dir, 'psnr.txt'), 'w') as f:
            f.write(f"Middle Frame PSNR: {pred['psnr']:.4f} dB\n")
    
    print(f"Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Jin et al. CVPR 2018 model on GoPro dataset')
    parser.add_argument('--gopro_path', type=str, required=True,
                       help='Path to GoPro test dataset (e.g., GOPRO_Large/test)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA (GPU)')
    parser.add_argument('--downsample', type=float, default=0.45,
                       help='Downsampling factor (default: 0.45 = 45%%)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.cuda and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU instead")
        args.cuda = False
    
    print(f"\n{'='*60}")
    print(f"GoPro Dataset Evaluation")
    print(f"{'='*60}")
    print(f"Dataset path: {args.gopro_path}")
    print(f"Downsampling factor: {args.downsample}")
    print(f"Using CUDA: {args.cuda}")
    print(f"{'='*60}\n")
    
    # Find all test sequences
    test_sequences = sorted(glob.glob(os.path.join(args.gopro_path, '*')))
    if not test_sequences:
        print(f"Error: No sequences found in {args.gopro_path}")
        print("Make sure the path points to GOPRO_Large/test/")
        return
    
    print(f"Found {len(test_sequences)} test sequences:")
    for seq in test_sequences:
        print(f"  - {os.path.basename(seq)}")
    
    # Generate all blurry test samples
    all_samples = []
    print(f"\nLoading blur and sharp images...")
    for seq_path in test_sequences:
        seq_name = os.path.basename(seq_path)
        
        blurs, sharps = load_gopro_sequence(seq_path)
        if len(blurs) == 0 or len(sharps) == 0:
            print(f"  Warning: No frames found in {seq_name}, skipping")
            continue
            
        print(f"  {seq_name}: {len(blurs)} blur images", end='')
        
        samples = create_blur_samples(blurs, sharps, args.downsample)
        print(f" -> {len(samples)} samples")
        
        all_samples.extend(samples)
    
    print(f"\nTotal test samples: {len(all_samples)}")
    print("Paper reports ~1100 samples from 11 sequences")
    
    if len(all_samples) == 0:
        print("Error: No test samples generated!")
        return
    
    # Load models
    models = load_models(cuda=args.cuda)
    
    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)
    results = evaluate_middle_frame(models, all_samples, cuda=args.cuda)
    
    # Save results
    save_results(results, args.output_dir)


if __name__ == '__main__':
    main()