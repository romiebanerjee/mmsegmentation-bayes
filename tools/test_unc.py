import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner
import argparse
import os
import torch


def parse_args():
    """
    Parse command line arguments for uncertainty testing.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    default_model_ckpt = "/path/to/default/checkpoint.pth"
    default_config = "/path/to/default/config.py"
    default_curvature_ckpt = "/path/to/default/kfac_state.pkl"
    default_inv_curvature_ckpt = "/path/to/default/kfac_state_inv.pkl"
    
    parser = argparse.ArgumentParser(
        description='Test uncertainty estimation for a segmentation model')
    
    parser.add_argument('--config', 
                      help='Config file path',
                      default=default_config)
    parser.add_argument('--ckpt', 
                      help='Checkpoint file path',
                      default=default_model_ckpt)
    parser.add_argument('--curvature_ckpt',  
                      help='Curvature state file path',
                      default=default_curvature_ckpt)
    parser.add_argument('--inv_curvature_ckpt',
                      help='Inverse curvature state file path',
                      default=default_inv_curvature_ckpt)
    parser.add_argument('--device', 
                      help='Device for model (e.g., "cuda:0", "cpu")',
                      default="cuda:0")
    parser.add_argument('--curvature_device',
                      help='Device for curvature calculations',
                      default=None)
    parser.add_argument('--save-gpu', 
                      help='Move tensors to CPU to save GPU memory',
                      action='store_true')
    parser.add_argument('--output-dir',
                      help='Directory to save results',
                      default='./work_dirs')
    parser.add_argument('--batch-size',
                      help='Batch size for testing',
                      type=int,
                      default=None)
    
    args = parser.parse_args()
    
    # Set curvature device to match model device if not specified
    if args.curvature_device is None:
        args.curvature_device = args.device
    
    return args


def setup_config(cfg, args):
    """
    Configure the runner settings based on command line arguments.
    
    Args:
        cfg (Config): Original configuration
        args (argparse.Namespace): Command line arguments
    
    Returns:
        Config: Modified configuration
    """
    # Set basic paths and devices
    cfg.work_dir = osp.join(args.output_dir, 
                           osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.ckpt
    cfg.device = args.device
    
    # Set curvature-related configurations
    cfg.curvature_device = args.curvature_device
    cfg.curvature_checkpoint = args.curvature_ckpt
    cfg.inverse_curvature_checkpoint = args.inv_curvature_ckpt
    cfg.save_gpu = args.save_gpu
    
    # Modify batch size if specified
    if args.batch_size is not None:
        cfg.test_dataloader.batch_size = args.batch_size
    
    return cfg


def verify_devices(args):
    """
    Verify that the specified devices are available.
    
    Args:
        args (argparse.Namespace): Command line arguments
    
    Raises:
        RuntimeError: If specified device is not available
    """
    if 'cuda' in args.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        device_idx = int(args.device.split(':')[-1])
        if device_idx >= torch.cuda.device_count():
            raise RuntimeError(f"Specified CUDA device {device_idx} not available")
    
    if 'cuda' in args.curvature_device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for curvature but not available")
        device_idx = int(args.curvature_device.split(':')[-1])
        if device_idx >= torch.cuda.device_count():
            raise RuntimeError(f"Specified CUDA device {device_idx} for curvature not available")


def main():
    """
    Main function to run uncertainty testing.
    """
    args = parse_args()
    
    # Verify devices before proceeding
    verify_devices(args)
    
    # Load and modify configuration
    cfg = Config.fromfile(args.config)
    cfg = setup_config(cfg, args)
    
    # Initialize and run testing
    runner = Runner.from_cfg(cfg)
    
    print('\n=== Device Configuration ===')
    print(f'MODEL DEVICE: {runner.device}')
    print(f'KFAC DEVICE: {runner.kfac.device}')
    print(f'Using {"GPU" if "cuda" in runner.device else "CPU"} for computations\n')
    
    # Run uncertainty testing
    runner.test_unc()


if __name__ == '__main__':
    main()