from typing import Dict
from pathlib import Path
import logging
import os
import json
import joblib
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor

from pytorch3d.transforms import matrix_to_quaternion
from datasets.dataset_meta import DATASETS_CONFIG
from datasets.base.scene_dataset import ModelType
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource, CameraData

from utils.geometry import (
    camera_model_rays,
    camera_shutter_fraction,
    interpolate_camera_poses,
    project_world_camera,
)

logger = logging.getLogger()

# define each class's node type
OBJECT_CLASS_NODE_MAPPING = {
    "Vehicle": ModelType.RigidNodes,
    "Pedestrian": ModelType.SMPLNodes,
    "Cyclist": ModelType.DeformableNodes
}
SMPLNODE_CLASSES = ["Pedestrian"]

# OpenCV to Dataset coordinate transformation
# opencv coordinate system: x right, y down, z front
# waymo coordinate system: x front, y left, z up
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

# Waymo Camera List:
# 0: front_camera
# 1: front_left_camera
# 2: front_right_camera
# 3: left_camera
# 4: right_camera
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4]

class WaymoCameraData(CameraData):
    def __init__(self, **kwargs):
        self._capture = None
        self._ray_cache = {}
        timing_path = (
            Path(kwargs['data_path'])
            / 'camera_timing'
            / f"{kwargs['start_timestep']:03d}_{kwargs['cam_id']}.json"
        )
        if timing_path.exists():
            resolution = json.loads(timing_path.read_text())['resolution']
            if len(resolution) != 2 or min(resolution) <= 0:
                raise ValueError('Invalid Waymo camera resolution')
            self.camera_metadata = {
                'camera_name': DATASETS_CONFIG['waymo'][kwargs['cam_id']]['camera_name'],
                'original_size': list(reversed(resolution)),
            }
        super().__init__(**kwargs)
        
    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # load camera intrinsics
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        # Native images use these coefficients in rays, projection and rendering.
        intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt")
        )
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array([k1, k2, p1, p2, k3])

        # load camera extrinsics
        cam_to_ego = np.loadtxt(
            os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        )
        # because we use opencv coordinate system to generate camera rays,
        # we need a transformation matrix to covnert rays from opencv coordinate
        # system to waymo coordinate system.
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            # transformation:
            #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self._capture = self._read_capture_poses(
            self.data_path, self.cam_id, self.start_timestep, self.end_timestep
        )
        if self._capture is not None:
            starts, ends, timings = self._capture
            if any(t['resolution'] != list(self.original_size[::-1]) for t in timings):
                raise ValueError('Waymo camera timing resolution does not match image metadata')
            self.cam_to_worlds = self._midpoint_poses(starts, ends)
        elif not self.undistort:
            raise ValueError(
                'Native Waymo needs camera timing/poses; rerun preprocessing with --process_keys pose'
            )

    @staticmethod
    def _read_capture_poses(data_path, cam_id, start_timestep, end_timestep):
        root = Path(data_path)
        if not (root / 'camera_timing').exists():
            return None
        anchor = np.linalg.inv(np.loadtxt(root / 'ego_pose' / f'{start_timestep:03d}.txt'))
        starts, ends, timings = [], [], []
        for t in range(start_timestep, end_timestep):
            stem = f'{t:03d}_{cam_id}'
            paths = [
                root / 'camera_pose_start' / (stem + '.txt'),
                root / 'camera_pose' / (stem + '.txt'),
                root / 'camera_timing' / (stem + '.json'),
            ]
            if not all(path.exists() for path in paths):
                raise ValueError(
                    'Incomplete Waymo camera timing/poses; rerun preprocessing with --process_keys pose'
                )
            starts.append(anchor @ np.loadtxt(paths[0]))
            ends.append(anchor @ np.loadtxt(paths[1]))
            timing = json.loads(paths[2].read_text())
            if (
                timing.get('pose_convention')
                != 'OpenCV camera-to-world at image-edge exposure centers'
            ):
                raise ValueError('Unknown Waymo camera timing convention; reprocess this scene')
            if timing.get('shutter_type') not in (
                'GLOBAL',
                'ROLLING_TOP_TO_BOTTOM',
                'ROLLING_BOTTOM_TO_TOP',
                'ROLLING_LEFT_TO_RIGHT',
                'ROLLING_RIGHT_TO_LEFT',
            ):
                raise ValueError('Unknown Waymo shutter direction')
            timings.append(timing)
        return (
            torch.from_numpy(np.stack(starts)).float(),
            torch.from_numpy(np.stack(ends)).float(),
            timings,
        )

    @staticmethod
    def _midpoint_poses(starts, ends):
        poses = []
        for start, end in zip(starts, ends):
            rotation, origin = interpolate_camera_poses(start, end, 0.5)
            pose = torch.eye(4, device=start.device, dtype=start.dtype)
            pose[:3, :3], pose[:3, 3] = rotation, origin
            poses.append(pose)
        return torch.stack(poses)

    def _capture_endpoints(self, frame_idx, width, height):
        if self.undistort or self._capture is None:
            pose = self.cam_to_worlds[frame_idx]
            return pose, pose, 'GLOBAL'
        starts, ends, timings = self._capture
        shutter = timings[frame_idx]['shutter_type']
        if shutter == 'GLOBAL':
            pose = self.cam_to_worlds[frame_idx]
            return pose, pose, shutter
        size = height if shutter in ('ROLLING_TOP_TO_BOTTOM', 'ROLLING_BOTTOM_TO_TOP') else width
        # Waymo timing is defined at image edges; gsplat endpoints refer to the
        # first/last pixel centers. Adjust at every training/render resolution.
        fraction = starts.new_tensor([0.5 / size, 1 - 0.5 / size])
        rotation, origin = interpolate_camera_poses(starts[frame_idx], ends[frame_idx], fraction)
        poses = torch.eye(4, device=starts.device).repeat(2, 1, 1)
        poses[:, :3, :3], poses[:, :3, 3] = rotation, origin
        return poses[0], poses[1], shutter

    def to(self, device):
        super().to(device)
        if self._capture is not None:
            start, end, timings = self._capture
            self._capture = start.to(device), end.to(device), timings
        self._ray_cache.clear()
        return self

    def get_image(self, frame_idx):
        image, camera = super().get_image(frame_idx)
        if self.undistort:
            return image, camera
        height, width = int(camera['height']), int(camera['width'])
        K = self.intrinsics[frame_idx].clone()
        K[0] *= width / self.WIDTH
        K[1] *= height / self.HEIGHT
        distortion = self.distortions[frame_idx]
        radial = torch.cat((distortion[[0, 1, 4]], distortion.new_zeros(3)))
        tangential = distortion[[2, 3]]
        key = height, width
        if key not in self._ray_cache:
            self._ray_cache[key] = camera_model_rays(
                height, width, K, 'pinhole', radial, tangential_coeffs=tangential
            )
        rays, valid = self._ray_cache[key]
        start, end, shutter = self._capture_endpoints(frame_idx, width, height)
        y, x = torch.meshgrid(
            torch.arange(height, device=K.device) + 0.5,
            torch.arange(width, device=K.device) + 0.5,
            indexing='ij',
        )
        fraction = camera_shutter_fraction(torch.stack((x, y), -1), width, height, shutter)
        rotation, origin = interpolate_camera_poses(start, end, fraction)
        image['origins'] = origin
        image['viewdirs'] = torch.nn.functional.normalize(
            torch.einsum('...ij,...j->...i', rotation, rays), dim=-1
        )
        image['direction_norm'] = 1 / rays[..., 2:].clamp_min(1e-8)
        invalid = ~valid
        if 'egocar_masks' in image:
            invalid = invalid | image['egocar_masks'].bool()
        image['egocar_masks'] = invalid.float()
        image['valid_masks'] = ~invalid
        camera.update(
            intrinsics=K,
            camera_model='pinhole',
            radial_coeffs=radial,
            tangential_coeffs=tangential,
            camera_to_world=start,
            camera_to_world_end=end,
            shutter_type=shutter,
        )
        return image, camera

    def project_points(self, points, frame_idx):
        start, end, shutter = self._capture_endpoints(frame_idx, self.WIDTH, self.HEIGHT)
        radial = tangential = None
        if not self.undistort:
            distortion = self.distortions[frame_idx]
            radial = torch.cat((distortion[[0, 1, 4]], distortion.new_zeros(3)))
            tangential = distortion[[2, 3]]
        pixels, depth, valid = project_world_camera(
            points,
            start,
            end,
            self.intrinsics[frame_idx],
            self.WIDTH,
            self.HEIGHT,
            'pinhole',
            radial_coeffs=radial,
            tangential_coeffs=tangential,
            shutter_type=shutter,
        )
        if self.egocar_mask is not None:
            selected = valid.nonzero().flatten()
            uv = pixels[selected].long()
            valid[selected] &= ~self.egocar_mask[uv[:, 1], uv[:, 0]].bool()
        return pixels, depth, valid

    @classmethod
    def get_camera2worlds(cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int) -> torch.Tensor:
        """
        Returns camera-to-world matrices for the specified camera and time range.

        Args:
            data_path (str): Path to the dataset.
            cam_id (str): Camera ID.
            start_timestep (int): Start timestep.
            end_timestep (int): End timestep.

        Returns:
            torch.Tensor: Camera-to-world matrices of shape (num_frames, 4, 4).
        """
        capture = cls._read_capture_poses(data_path, cam_id, start_timestep, end_timestep)
        if capture is not None:
            return cls._midpoint_poses(capture[0], capture[1])
        # Load camera extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_path, "extrinsics", f"{cam_id}.txt"))
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # Load ego poses and compute camera-to-world matrices
        cam_to_worlds = []
        ego_to_world_start = np.loadtxt(os.path.join(data_path, "ego_pose", f"{start_timestep:03d}.txt"))
        
        for t in range(start_timestep, end_timestep):
            ego_to_world_current = np.loadtxt(os.path.join(data_path, "ego_pose", f"{t:03d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)

        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

class WaymoPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()

        for idx, cam_id in enumerate(self.camera_list):
            logger.info(f"Loading camera {cam_id}")
            camera = WaymoCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                viewer=self.data_cfg.get("viewer", False),
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(
                unique_cam_idx = idx,
                unique_img_idx = unique_img_idx
            )
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera
    
    def load_objects(self):
        """
        get ground truth bounding boxes of the dynamic objects

        instances_info = {
            "0": # simplified instance id
                {
                    "id": str,
                    "class_name": str,
                    "frame_annotations": {
                        "frame_idx": List,
                        "obj_to_world": List,
                        "box_size": List,
                },
            ...
        }
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """
        instances_info_path = os.path.join(self.data_path, "instances", "instances_info.json")
        frame_instances_path = os.path.join(self.data_path, "instances", "frame_instances.json")
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)
        instances_model_types = np.ones(num_instances) * -1
        
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for k, v in instances_info.items():
            instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
            for frame_idx, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):
                # the first ego pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
                instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
                instances_size[frame_idx, int(k)] = np.array(box_size)
        
        # get frame valid instances
        # shape (num_frames, num_instances)
        per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
        for frame_idx, valid_instances in frame_instances.items():
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1
        
        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(instances_pose[self.start_timestep:self.end_timestep]).float()
        instances_size = torch.from_numpy(instances_size[self.start_timestep:self.end_timestep]).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(per_frame_instance_mask[self.start_timestep:self.end_timestep]).bool()
        
        # filter out the instances that are not visible in selected frames
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]
        
        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose
        # (num_instances, 3)
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask
        # (num_instances)
        self.instances_true_id = instances_true_id
        # (num_instances)
        self.instances_model_types = instances_model_types
        
        if self.data_cfg.load_smpl:
            # Collect camera-to-world matrices for all available cameras
            cam_to_worlds = {}
            for cam_id in AVAILABLE_CAM_LIST:
                cam_to_worlds[cam_id] = WaymoCameraData.get_camera2worlds(
                    self.data_path, 
                    str(cam_id), 
                    self.start_timestep, 
                    self.end_timestep
                )

            # load SMPL parameters
            smpl_dict = joblib.load(os.path.join(self.data_path, "humanpose", "smpl.pkl"))
            frame_num = self.end_timestep - self.start_timestep
            
            smpl_human_all = {}
            for fi in tqdm(range(self.start_timestep, self.end_timestep), desc="Loading SMPL"):
                for instance_id, ins_smpl in smpl_dict.items():
                    if instance_id not in smpl_human_all:
                        smpl_human_all[instance_id] = {
                            "smpl_quats": torch.zeros((frame_num, 24, 4), dtype=torch.float32),
                            "smpl_trans": torch.zeros((frame_num, 3), dtype=torch.float32),
                            "smpl_betas": torch.zeros((frame_num, 10), dtype=torch.float32),
                            "frame_valid": torch.zeros((frame_num), dtype=torch.bool)
                        }
                        smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0
                    if ins_smpl["valid_mask"][fi]:
                        betas = ins_smpl["smpl"]["betas"][fi]
                        smpl_human_all[instance_id]["smpl_betas"][fi - self.start_timestep] = betas
                        
                        body_pose = ins_smpl["smpl"]["body_pose"][fi]
                        smpl_orient = ins_smpl["smpl"]["global_orient"][fi]
                        cam_depend = ins_smpl["selected_cam_idx"][fi].item()
                        
                        c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
                        world_orient = c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()
                        smpl_quats = matrix_to_quaternion(
                            torch.cat([world_orient[None, ...], body_pose], dim=0)
                        )
                        
                        ii = instances_info[str(instance_id)]['frame_annotations']["frame_idx"].index(fi)
                        o2w = np.array(
                            instances_info[str(instance_id)]['frame_annotations']["obj_to_world"][ii]
                        )
                        o2w = torch.from_numpy(
                            np.linalg.inv(ego_to_world_start) @ o2w
                        )
                        # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]
                        
                        smpl_human_all[instance_id]["smpl_quats"][fi - self.start_timestep] = smpl_quats
                        smpl_human_all[instance_id]["smpl_trans"][fi - self.start_timestep] = o2w[:3, 3]
                        smpl_human_all[instance_id]["frame_valid"][fi - self.start_timestep] = True

            self.smpl_human_all = smpl_human_all
            
class WaymoLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, "lidar", f"{t:03d}.bin")
            )
        self.lidar_filepaths = np.array(lidar_filepaths)

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        # Note that in the Waymo Open Dataset, the lidar coordinate system is the same
        # as the vehicle coordinate system
        lidar_to_worlds = []

        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(
            np.stack(lidar_to_worlds, axis=0)
        ).float()

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        # flow/ground info are used for evaluation only
        flows, flow_classes, grounds = [], [], []
        # in waymo, we simplify timestamps as the time indices
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):
            # each lidar_info contains an Nx14 array
            # from left to right:
            # origins: 3d, points: 3d, flows: 3d, flow_class: 1d,
            # ground_labels: 1d, intensities: 1d, elongations: 1d, laser_ids: 1d
            lidar_info = np.memmap(
                self.lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length

            # select lidar points based on the laser id
            if self.data_cfg.only_use_top_lidar:
                # laser_ids: 0: TOP, 1: FRONT, 2: SIDE_LEFT, 3: SIDE_RIGHT, 4: REAR
                lidar_info = lidar_info[lidar_info[:, 13] == 0]

            lidar_origins = torch.from_numpy(lidar_info[:, :3]).float()
            lidar_points = torch.from_numpy(lidar_info[:, 3:6]).float()
            lidar_ids = torch.from_numpy(lidar_info[:, 13]).float()
            lidar_flows = torch.from_numpy(lidar_info[:, 6:9]).float()
            lidar_flow_classes = torch.from_numpy(lidar_info[:, 9]).long()
            ground_labels = torch.from_numpy(lidar_info[:, 10]).long()
            # we don't collect intensities and elongations for now

            # select lidar points based on a truncated ego-forward-directional range
            # this is to make sure most of the lidar points are within the range of the camera
            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (
                    lidar_points[:, 0] > self.data_cfg.truncated_min_range
                )
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            lidar_flows = lidar_flows[valid_mask]
            lidar_flow_classes = lidar_flow_classes[valid_mask]
            ground_labels = ground_labels[valid_mask]
            # transform lidar points from lidar coordinate system to world coordinate system
            lidar_origins = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            # scene flows are in the lidar coordinate system, so we need to rotate them
            lidar_flows = (self.lidar_to_worlds[t][:3, :3] @ lidar_flows.T).T
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            # we use time indices as the timestamp for waymo dataset
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)
            flows.append(lidar_flows)
            flow_classes.append(lidar_flow_classes)
            grounds.append(ground_labels)
            # we use time indices as the timestamp for waymo dataset
            timesteps.append(lidar_timestamp)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )
        logger.info("Filter condition:")
        logger.info(f"  only_use_top_lidar: {self.data_cfg.only_use_top_lidar}")
        logger.info(f"  truncated_max_range: {self.data_cfg.truncated_max_range}")
        logger.info(f"  truncated_min_range: {self.data_cfg.truncated_min_range}")

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)
        # becasue the flows here are velocities (m/s), and the fps of the lidar is 10,
        # we need to divide the velocities by 10 to get the displacements/flows
        # between two consecutive lidar scans
        self.flows = torch.cat(flows, dim=0) / 10.0
        self.flow_classes = torch.cat(flow_classes, dim=0)
        self.grounds = torch.cat(grounds, dim=0).bool()

        # the underscore here is important.
        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()

    def to(self, device: torch.device):
        super().to(device)
        self.flows = self.flows.to(device)
        self.flow_classes = self.flow_classes.to(device)
        self.grounds = self.grounds.to(self.device)

    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        flows = self.flows[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
            "lidar_flows": flows,
        }

    def delete_invisible_pts(self) -> None:
        """
        Clear the unvisible points.
        """
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self.flows = self.flows[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
        else:
            logger.info("[Lidar] No unvisible points to clear.")