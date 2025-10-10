# coding=utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
from vggt.models.vggt import VGGT
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from ba import run_vggt_with_ba
import argparse
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

from quarot.utils import VggtQuantModel, set_ignore_quantize,load_qs_parameters
from quarot.args_utils import get_config

import time
# from compare import compare_models,compare_model_structure,print_model_comparison_summary,verify_models_identical

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False


def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3D rotation matrix in Point3D format
        Trans: 3D translation vector in Point3D format

    Returns:
        extri_opencv: 3x4 extrinsic matrix in OpenCV format
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def align_to_first_camera(camera_poses):
    """
    Align all camera poses to the first camera's coordinate frame.

    Args:
        camera_poses: Tensor of shape (N, 4, 4) containing camera poses as SE3 transformations

    Returns:
        Tensor of shape (N, 4, 4) containing aligned camera poses
    """
    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = torch.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses

def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on CO3D dataset')
    parser.add_argument('--debug_mode', type=str,default='all', help='Enable debug mode (only test on specific category)')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--fast_eval', action='store_true', default=False, help='Only evaluate 10 sequences per category')
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str,required=True, help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, help='Path to CO3D annotations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')

    parser.add_argument('--dtype', type=str, default='quantvggt_w6a6', help='Data type for model inference')
    parser.add_argument('--each_nsamples', type=int, default='2', help='Number of samples per class')
    parser.add_argument('--not_smooth', action='store_true', help='')
    parser.add_argument('--not_rot', action='store_true', help='')
    parser.add_argument('--lwc', action='store_true', help='lwc')
    parser.add_argument('--lac', action='store_true', help='lac')
    parser.add_argument('--rv', action='store_true', help='rot then smooth')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--cache_path', type=str, default=None, help='cache path')
 
    parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
    parser.add_argument('--fuse_qkv', action='store_true', default=False, help='Fuse QKV projections')
    parser.add_argument('--resume_qs', action='store_true', default=False, help='Resume SmoothQuant calibration')
    parser.add_argument('--use_gptq', action='store_true', default=False, help='Use GPTQ quantization')
    parser.add_argument('--resume_gptq', action='store_true', default=False, help='Resume GPTQ quantization')

    return parser.parse_args()



def get_simple_calibration_data( device, min_num_images,num_frames,
    co3d_dir, co3d_anno_dir, SEEN_CATEGORIES, each_nsamples=5, cache_path=None):
  
    total_num = 0
    calib_data = []
    for category in SEEN_CATEGORIES:
        print(f"Loading calibration annotation for {category} test set")
        annotation_file = os.path.join(co3d_anno_dir, f"{category}_test.jgz")

        print(f"annotation_file: {annotation_file}")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        seq_names = sorted(list(annotation.keys()))
        seq_names = random.sample(seq_names, min(each_nsamples, len(seq_names))) 
        total_num += min(each_nsamples, len(seq_names))
        print(f"Processing Sequences: {seq_names}")

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if len(seq_data) < min_num_images:  # Ensure sufficient data
                continue
            metadata = []
            for data in seq_data:
      
                if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                    continue
                extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
                metadata.append({
                    "filepath": data["filepath"],
                    "extri": extri_opencv,
                })

            ids = np.random.choice(len(metadata), num_frames, replace=False) # 只是索引 
            image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
            images = load_and_preprocess_images(image_names).to(device)

            input_dict = {}
            input_dict["images"] = images
            input_dict["category"] = category # 新增类别
            input_dict["seq_name"] = seq_name
            calib_data.append(input_dict)
 
    print("cache_path",{cache_path},"total_num",total_num)
    if cache_path:
        print(f"Saving calibration data to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(calib_data, cache_path)

    return calib_data,len(calib_data)


def print_model_modules(model):
    for name, module in model.named_modules():
        print(f"module_name: {name}: {type(module)}")

def get_module_info(module, prefix=''):
    info = {
        'name': prefix,
        'type': str(type(module)),
        'param_count': sum(p.numel() for p in module.parameters() if p.requires_grad),
        'children': []
    }

    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        child_info = get_module_info(child, child_prefix)
        info['children'].append(child_info)
    
    return info

def save_model_structure_to_json(model, output_path):
    """将模型结构保存为JSON文件"""
    model_info = get_module_info(model)
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"模型结构已保存到 {output_path}")
    exit()

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_sequence(model, seq_name, seq_data, category, co3d_dir, min_num_images, num_frames, use_ba, device, dtype):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:

        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    ids = np.random.choice(len(metadata), num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    images = load_and_preprocess_images(image_names).to(device)

    if use_ba:
        try:
            pred_extrinsic = run_vggt_with_ba(model, images, image_names=image_names, dtype=dtype)
        except Exception as e:
            print(f"BA failed with error: {e}. Falling back to standard VGGT inference.")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            with torch.cuda.amp.autocast(dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                pred_extrinsic = extrinsic[0]
    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                
                predictions = model(images)

        with torch.cuda.amp.autocast(dtype=torch.float64):
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            pred_extrinsic = extrinsic[0]


    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)


        pred_se3 = align_to_first_camera(pred_se3)
        gt_se3 = align_to_first_camera(gt_se3)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()
    

def load_model(device, each_nsamples,min_num_images, num_frames,category,co3d_anno_dir,co3d_dir,
               model_path, dtype, compile, fuse_qkv, resume_qs, use_gptq, resume_gptq,debug_mode,
               not_smooth, not_rot, lwc,lac,rv,exp_name,cache_path = None):
  
    print("Initializing and loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    calib_data = None
    if dtype in['raw']:
        if os.path.exists(cache_path):
            print(f" find calib file: {cache_path}")
            calib_data = torch.load(cache_path)

        elif not os.path.exists(cache_path):
            print(f" generate calib file,each class: {each_nsamples}")
            calib_data, calib_data_num = get_simple_calibration_data(device, min_num_images,num_frames,
                                                co3d_dir,co3d_anno_dir, category, each_nsamples, cache_path=cache_path)
            
        return model,calib_data


    if dtype in ["quantvggt_w4a4","quantvggt_w6a6","quantvggt_w8a8"]:
        import re
        wbit = re.search(r'w(\d+)', dtype)
        abit = re.search(r'a(\d+)', dtype)

        if wbit:
            wbit = int(wbit.group(1))
        if abit:
            abit = int(abit.group(1))


        config = get_config()
        config.update_from_args(wbit=wbit, abit=abit, not_smooth=not_smooth, not_rot=not_rot, lwc=lwc, lac=lac, rv=rv, model_id=model_path, exp_name=exp_name)
        print(f"wbit:{wbit},abit:{abit},not_rot:{config.not_rot},not_smooth:{config.not_smooth},lwc:{config.lwc},lac:{config.lac},rv:{config.rv}")
       
        if cache_path is not None:
            cache_path = cache_path
        else:
            cache_path = f"{config.cache_dir}/{model_path.replace('/', '_')}_{debug_mode}_{each_nsamples}_calib_data.pt"
        print("exp_dir:",config.exp_dir)
        print("cache_dir:",config.cache_dir)
        print("cache_path:",cache_path)

        if os.path.exists(cache_path) and not resume_qs:
            calib_data = torch.load(cache_path)

            config.update_nsamples(len(calib_data))
            print(f"get calib file,nsamples = {len(calib_data)} == {config.nsamples}")
        
        elif not os.path.exists(cache_path) and not resume_qs :
            calib_data, calib_data_num = get_simple_calibration_data(device, min_num_images,num_frames,
                                                 co3d_dir,co3d_anno_dir, category, each_nsamples, cache_path=cache_path)

            config.update_nsamples(calib_data_num)
            print(f"fet calib file ,nsamples = {len(calib_data)} == {config.nsamples}")
        else:
            print("Non't need calib file")
            calib_data = None

        VggtQuantModel(config,model,calib_data, wbit=wbit, abit=abit,
                            resume_qs=resume_qs, use_gptq=use_gptq, resume_gptq=resume_gptq,
                            model_id=model_path, exp_name=exp_name)
   
        model.to(device)

    if compile:
        model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    
    return model,calib_data

def main():
    args = setup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    SEEN_CATEGORIES = ["apple"]
    if args.debug_mode == "apple":
        SEEN_CATEGORIES = ["apple"]
    elif args.debug_mode == "five":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag"]
    elif args.debug_mode == "more":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag","carrot","cellphone", "motorcycle","umbrella","toaster"]
    elif args.debug_mode == "all":
        SEEN_CATEGORIES = [
        "apple", "backpack", "banana", "baseballbat", "baseballglove",
        "bench", "bicycle", "bottle", "bowl", "broccoli",
        "cake", "car", "carrot", "cellphone", "chair",
        "cup", "donut", "hairdryer", "handbag", "hydrant",
        "keyboard", "laptop", "microwave", "motorcycle", "mouse",
        "orange", "parkingmeter", "pizza", "plant", "stopsign",
        "teddybear", "toaster", "toilet", "toybus", "toyplane",
        "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
    ]


    model , _ = load_model(device,args.each_nsamples,
                       min_num_images =args.min_num_images, num_frames =  args.num_frames,category =SEEN_CATEGORIES,co3d_anno_dir = args.co3d_anno_dir,co3d_dir = args.co3d_dir, # 这两个参数用于calib_data
                        model_path=args.model_path, dtype=args.dtype,
                        compile= args.compile, fuse_qkv=args.fuse_qkv,
                        resume_qs=args.resume_qs, use_gptq=args.use_gptq,
                        resume_gptq=args.resume_gptq,debug_mode=args.debug_mode,
                        not_smooth=args.not_smooth, not_rot=args.not_rot,lac=args.lac,lwc=args.lwc,rv=args.rv,exp_name=args.exp_name,cache_path=args.cache_path)

    print(SEEN_CATEGORIES)
    set_random_seeds(args.seed)

    per_category_results = {}

    for category in SEEN_CATEGORIES:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        rError = []
        tError = []

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names)>=10:
            seq_names = seq_names[:10] 
        seq_names = sorted(seq_names)

        print("Testing Sequences: ")
        print(seq_names)

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")

            if args.debug_mode and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue
            
            seq_rError, seq_tError = process_sequence(
                model, seq_name, seq_data, category, args.co3d_dir,
                args.min_num_images, args.num_frames, args.use_ba, device, dtype,
            )   
            print("-" * 50)
            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError) # 
                tError.extend(seq_tError)

        if not rError:
            print(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        print("="*80)

        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {category} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        print("="*80)

    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    mean_AUC_30=0.0
    for category in sorted(per_category_results.keys()):
        print(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30), {per_category_results[category]['Auc_15']:.4f} (AUC@15), {per_category_results[category]['Auc_5']:.4f} (AUC@5), {per_category_results[category]['Auc_3']:.4f} (AUC@3)")
    mean_AUC_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
    mean_AUC_15 = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
    mean_AUC_5 = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
    mean_AUC_3 = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
    print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")

    return 
if __name__ == "__main__":
    main()

