import matplotlib.pyplot as plt
from mmseg.apis import MCGLM
import argparse
import os.path as osp
from typing import List, Optional

class SegmentationUncertaintyDemo:
    """Demonstration of segmentation uncertainty using MCGLM"""
    
    def __init__(self, 
                model_ckpt: str,
                config: str,
                device: str,
                curvature_ckpt: str,
                inv_curvature_ckpt: str):
        """
        Initialize the demo with model and uncertainty estimation configuration.
        
        Args:
            model_ckpt: Path to model checkpoint
            config: Path to model config file
            device: Device to run on (e.g., 'cuda:0')
            curvature_ckpt: Path to K-FAC curvature checkpoint
            inv_curvature_ckpt: Path to inverse curvature checkpoint
        """
        self.mcglm = MCGLM(
            model_checkpoint=model_ckpt,
            config=config,
            device=device,
            curvature_checkpoint=curvature_ckpt,
            inverse_curvature_checkpoint=inv_curvature_ckpt
        )
        self.device = device

    def run_demo(self,
                image_path: str,
                output_path: str,
                eps: float = 1e-6,
                iters: int = 1,
                eig_idxs: Optional[List[float]] = None,
                show_results: bool = False):
        """
        Run uncertainty estimation demo on an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save results
            eps: Perturbation scale for uncertainty estimation
            iters: Number of Monte Carlo iterations
            eig_idxs: List of eigen-indices to test (None for default range)
            show_results: Whether to display results interactively
        """
        if eig_idxs is None:
            eig_idxs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # Default eigen-index range
            
        # Create figure for visualization
        fig, axs = plt.subplots(
            len(eig_idxs), 
            figsize=(10, 5*len(eig_idxs)),
            squeeze=False
        )
        fig.suptitle('Segmentation Uncertainty Across Eigen-Indices', y=0.92)
        
        # Calculate and visualize uncertainty for each eigen-index
        for i, eig_idx in enumerate(eig_idxs):
            seg_unc = self.mcglm.pred_seg_logits_uncertainty(
                image_path,
                eps=eps,
                eig_idx=eig_idx,
                iters=iters
            )
            
            # Visualize results
            im = axs[i,0].imshow(seg_unc.cpu().numpy())
            axs[i,0].set_title(f'Eigen-Index: {eig_idx:.2f}')
            plt.colorbar(im, ax=axs[i,0])
        
        # Save and optionally show results
        plt.tight_layout()
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Results saved to {output_path}")
        
        if show_results:
            plt.show()
        plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Segmentation Uncertainty Demo')
    
    # Model configuration
    parser.add_argument('--config', 
                       default="/path/to/default/config.py",
                       help='Model config file path')
    parser.add_argument('--ckpt',
                       default="/path/to/default/checkpoint.pth",
                       help='Model checkpoint path')
    parser.add_argument('--curvature-ckpt',
                       default="/path/to/default/kfac_state.pkl",
                       help='K-FAC curvature checkpoint path')
    parser.add_argument('--inv-curvature-ckpt',
                       default="/path/to/default/kfac_state_inv.pkl",
                       help='Inverse curvature checkpoint path')
    
    # Runtime configuration
    parser.add_argument('--device',
                       default="cuda:0",
                       help='Device to use (e.g., "cuda:0", "cpu")')
    parser.add_argument('--image',
                       default="/path/to/default/demo.png",
                       help='Input image path')
    parser.add_argument('--output',
                       default="/path/to/default/seg_uncs.png",
                       help='Output visualization path')
    
    # Uncertainty estimation parameters
    parser.add_argument('--eps',
                       type=float,
                       default=1e-6,
                       help='Perturbation scale for uncertainty estimation')
    parser.add_argument('--iters',
                       type=int,
                       default=1,
                       help='Number of Monte Carlo iterations')
    parser.add_argument('--show',
                       action='store_true',
                       help='Show results interactively')
    
    return parser.parse_args()


def main():
    """Main function to run the demo"""
    args = parse_args()
    
    # Initialize demo
    demo = SegmentationUncertaintyDemo(
        model_ckpt=args.ckpt,
        config=args.config,
        device=args.device,
        curvature_ckpt=args.curvature_ckpt,
        inv_curvature_ckpt=args.inv_curvature_ckpt
    )
    
    # Run demo with specified parameters
    demo.run_demo(
        image_path=args.image,
        output_path=args.output,
        eps=args.eps,
        iters=args.iters,
        show_results=args.show
    )


if __name__ == '__main__':
    main()