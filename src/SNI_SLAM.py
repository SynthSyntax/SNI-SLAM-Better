# This file is a part of SNI-SLAM

import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

from src.networks.model_manager import ModelManager

torch.multiprocessing.set_sharing_strategy('file_system')

import wandb


class SNI_SLAM():
    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']
        self.device = cfg['device']
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        self.scale = cfg['scale']
        self.load_bound(cfg)

        model = config.get_model(cfg, self.bound)
        self.shared_decoders = model

        self.enable_wandb = cfg['func']['enable_wandb']
        if self.enable_wandb:
            self.wandb_run = wandb.init(project="sni_slam")

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = (torch.zeros
                            ((self.n_img, 4, 4)))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()

        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()

        self.model_manager = ModelManager(cfg)
        self.model_manager.get_share_memory()

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """Load scene bound from config (scaled). Hash grids normalize to this bound."""
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound']) * self.scale).float()

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def load_ckpt(self, ckpt_path):
        """Load checkpoint and restore decoder (hash grids live inside), poses, keyframe list."""
        print(f'Loading checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])
        self.estimate_c2w_list[:] = ckpt['estimate_c2w_list']
        return ckpt['keyframe_list'], ckpt['idx']

    def run_mesh_only(self):
        """Load latest checkpoint and generate mesh without re-running tracking/mapping."""
        ckpts = sorted(os.listdir(self.ckptsdir))
        assert len(ckpts) > 0, f'No checkpoints found in {self.ckptsdir}'
        ckpt_path = os.path.join(self.ckptsdir, ckpts[-1])
        keyframe_list, idx = self.load_ckpt(ckpt_path)

        # Reconstruct keyframe_dict from disk using saved keyframe indices
        keyframe_dict = []
        for ki in keyframe_list:
            _, gt_color, gt_depth, gt_c2w, _ = self.frame_reader[ki]
            keyframe_dict.append({
                'idx': ki,
                'gt_c2w': gt_c2w.to(self.device),
                'color': gt_color.to(self.device),
                'depth': gt_depth.to(self.device),
                'est_c2w': self.estimate_c2w_list[ki].clone(),
            })

        print(f'Generating mesh from checkpoint at idx {idx}...')
        mesh_out_semantic = f'{self.output}/mesh/final_mesh_semantic.ply'
        mesh_out_color = f'{self.output}/mesh/final_mesh_color.ply'
        self.mesher.get_mesh(mesh_out_color, self.shared_decoders, keyframe_dict,
                             self.device, mesh_out_semantic=mesh_out_semantic, semantic=False)
        from src.tools.cull_mesh import cull_mesh
        cull_mesh(mesh_out_color, self.cfg, self.args, self.device,
                  estimate_c2w_list=self.estimate_c2w_list)

    def run(self):
        """
        Dispatch Threads.
        """

        if self.args.mesh_only:
            self.run_mesh_only()
            return

        processes = []
        for rank in range(0, 2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))

            p.start()
            processes.append(p)
        for p in processes:
            p.join()

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
