import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import re

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class ReplicaDataset(GradSLAMDataset):
    @staticmethod
    def _extract_frame_id(path: str) -> Optional[int]:
        stem = Path(path).stem
        match = re.search(r"(\d+)$", stem)
        if match is None:
            return None
        return int(match.group(1))

    def _build_frame_map(self, paths: List[str]) -> Dict[int, str]:
        frame_map: Dict[int, str] = {}
        for path in paths:
            frame_id = self._extract_frame_id(path)
            if frame_id is not None:
                frame_map[frame_id] = path
        return frame_map

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_all = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_all = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        semantic_all = natsorted(glob.glob(f"{self.input_folder}/semantic_remap/*.png"))

        color_map = self._build_frame_map(color_all)
        depth_map = self._build_frame_map(depth_all)
        semantic_map = self._build_frame_map(semantic_all)

        common_ids = sorted(set(color_map) & set(depth_map) & set(semantic_map))
        self.frame_ids = common_ids

        color_paths = [color_map[idx] for idx in common_ids]
        depth_paths = [depth_map[idx] for idx in common_ids]
        semantic_paths = [semantic_map[idx] for idx in common_ids]

        embedding_paths = None
        if self.load_embeddings:
            embedding_all = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
            embedding_map = self._build_frame_map(embedding_all)
            valid_ids = [idx for idx in common_ids if idx in embedding_map]
            self.frame_ids = valid_ids
            color_paths = [color_map[idx] for idx in valid_ids]
            depth_paths = [depth_map[idx] for idx in valid_ids]
            semantic_paths = [semantic_map[idx] for idx in valid_ids]
            embedding_paths = [embedding_map[idx] for idx in valid_ids]

        return color_paths, depth_paths, semantic_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        frame_ids = getattr(self, "frame_ids", None)
        if not frame_ids:
            frame_ids = list(range(self.num_imgs))
        for frame_id in frame_ids[: self.num_imgs]:
            if frame_id >= len(lines):
                break
            line = lines[frame_id]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
