from os import path as osp
import pickle
from typing import Union, List, Any, Dict, Optional
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmengine.curvature import KFAC


class MCGLM():
    """
    Base class to initiate a GLM (Generalized Linear Model) based on a pre-trained segementor model.
    A GLM of a pre-trained model f(theta, x) is the linar approximation of f at theta = pre-trained weights, and evaluated at a given x.
    This is obtained in principle by taking the Jacobian of f at the pre-trained point

    Args:
        confg (str): path to model config file
        model_checkpoint (str): path to pre-trained model location
        device (torch.device): cuda device to load model
        curvature checkpoint (str): path to pre-trained model KFAC state dictionary
        mc_iters (int): number of iterations used in monte-carlo approimation of Jacobian

    """
    def __init__(self, 
                model_checkpoint:str, 
                config:str, 
                device:str, 
                curvature_checkpoint:str,
                inverse_curvature_checkpoint: Union[str, None]
                ):
        self.n_labels = None   
        print('Loading base model ... ')     
        self.model = init_model(config, model_checkpoint, device=device)
        self.curvature_checkpoint = curvature_checkpoint
        self.inv_curvature_checkpoint = inverse_curvature_checkpoint
        self.device = device
        self._init_kfac()


    def _init_kfac(self):

        """
        Load the KFAC object on GPU
        """

        print('Initiating KFAC object ...')
        self.kfac = KFAC(model = self.model, device = self.device)

        print('Loading KFAC object from: ', self.curvature_checkpoint)
        with open(self.curvature_checkpoint, 'rb') as f:
            state_dict = pickle.load(f)

        if self.inv_curvature_checkpoint:
            print("Inverse KFAC state found: ", self.inv_curvature_checkpoint)
            with open(self.inv_curvature_checkpoint, 'rb') as f:
                contents = pickle.load(f)

        
            if isinstance(contents, dict):
                inv_state_dict = contents
                eig_vals = None
                eig_vecs = None


            elif isinstance(contents, list):
                if len(contents) == 1:
                    inv_state_dict = contents[0]
                    eig_vals = None
                    eig_vecs = None
                else:
                    assert len(contents)==3, "expecting 3 dictionaries"
                    print("KFAC eigenspaces found")
                    inv_state_dict, eig_vals, eig_vecs = contents
            else:
                raise ValueError("Expecting dict or list")

        else:
            print("Inverse KFAC not found, calculating inverse ...")
            inv_state_dict = None
            eig_vals = None
            eig_vecs = None
        self.kfac = self.kfac.load(state_dict, inv_state_dict, eig_vals, eig_vecs)

    def pred_seg_logits_mc(self, image, eps, eig_idx, iters):
        """produce logits from several monte carlo weight samples"""
        result = inference_model(self.model, image)
        pred_seg_logits = result.seg_logits.data #pred_logits.shape = (19,512,1024), 19 = no.of classes
        mc_preds = []

        with torch.no_grad():
            for i in range(iters):
                print('sampling range ', i)
                self.kfac.sample_and_replace(eigen_index = eig_idx, eps=eps)
                result = inference_model(self.model, image)
                mc_preds.append(result.seg_logits.data)
        pred_seg_logits_stacked = torch.stack(mc_preds)  #pred_logits_stacked.shape = (samples,19,512,1024), 19 = no.of classes
        return pred_seg_logits, pred_seg_logits_stacked


    def pred_seg_logits_uncertainty(self, image, eps, eig_idx, iters):
        """Use monte-carlo predictions to deserve GLM uncertainty"""
        pred_seg_logits, pred_seg_logits_stacked = self.pred_seg_logits_mc(image, eps, eig_idx, iters)
        A = (pred_seg_logits_stacked - pred_seg_logits)/eps
        B = torch.matmul(A.permute(2,3,1,0), A.permute(2,3,0,1)) #B.shape = (512,1024,19,19)
        C = B.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) #C.shape = (512,1024)
        return C


    def pred_seg_logits_unc_graded(self, image, eps, eig_idx, iters):
        pred_seg_logits, pred_seg_logits_stacked = self.pred_seg_logits_mc(image, eps, eig_idx, iters)
        A = (pred_seg_logits_stacked - pred_seg_logits)/eps
        C_graded = []

        for i in range(iters):
            B = torch.matmul(A[[i], :,:,:].permute(2,3,1,0), A[[i],:,:,:].permute(2,3,0,1)) #B.shape = (512,1024,19,19)
            C = B.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) #C.shape = (512,1024)
            C_graded.append(C)

        return C_graded











        
