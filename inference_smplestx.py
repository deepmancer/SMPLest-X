# Standard library imports
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
import pickle

# Third-party imports
import cv2
import numpy as np
import smplx
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm
from ultralytics import YOLO

# Local application imports
from human_models.human_models_smplerx import SMPLX
from main.base import Tester
from main.config import Config
from smplestx_utils.data_utils import load_img, process_bbox, generate_patch_image
from smplestx_utils.visualization_utils import render_mesh


def save_smplx_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: os.PathLike | str) -> None:
    """Persist SMPL-X vertices + triangular faces to an OBJ file."""
    if vertices is None or faces is None:
        return

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N, 3); got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with shape (M, 3); got {faces.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as obj_file:
        for vx, vy, vz in vertices:
            obj_file.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for f0, f1, f2 in faces:
            obj_file.write(f"f {f0 + 1} {f1 + 1} {f2 + 1}\n")


def get_bust_asset_paths(bust_assets_dir):
    return {
        'flame_embed_68': 'assets/body_models/landmarks/flame/flame_static_embedding_68_v4.npz',
        'flame_smplx_corr': "assets/body_models/base_models/smplx/vertex_mappings/smplx_flame_vertex_ids.npy",
        'flame_generic_model': 'assets/body_models/base_models/flame/parametric_models/generic_model.pkl',
    }

def validate_asset_paths(asset_paths):
    missing_files = []
    for asset_name, path in asset_paths.items():
        if asset_name == 'smplx_models_dir':
            if not os.path.isdir(path):
                missing_files.append(f"{asset_name}: {path} (directory)")
        else:
            if not os.path.exists(path):
                missing_files.append(f"{asset_name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing required asset files:\n" + "\n".join(missing_files))
    
    return True


class YOLOManager:
    def __init__(self, model_path=None, use_yolo=False, cfg=None):
        self.use_yolo = use_yolo
        self.detector = None
        self.cfg = cfg
        
        if self.use_yolo:
            self._initialize_detector(model_path)
    
    def _initialize_detector(self, model_path=None):
        try:
            model_path = 'modules/SMPLestX/pretrained_models/yolov8x.pt'
            self.detector = YOLO(model_path)
            print(f"YOLO detector initialized with model: {model_path}")
        except Exception as e:
            print(f"Error initializing YOLO detector: {e}")
            self.use_yolo = False
            self.detector = None
    
    def detect_person(self, image):
        if not self.use_yolo or self.detector is None:
            # Return full image bbox when YOLO is disabled
            h, w = image.shape[:2]
            bbox_xyxy = np.array([0, 0, w, h])
            bbox_xywh = np.array([0, 0, w, h])
            return bbox_xyxy, bbox_xywh
        
        try:
            # Get detection settings from config or use defaults
            device = 'cuda'
            classes = 0  # person class
            conf = getattr(self.cfg.inference.detection, "conf", 0.5) if self.cfg else 0.5
            save = getattr(self.cfg.inference.detection, "save", False) if self.cfg else False
            verbose = getattr(self.cfg.inference.detection, "verbose", False) if self.cfg else False
            
            yolo_bbox = self.detector.predict(
                image,
                device=device,
                classes=classes,
                conf=conf,
                save=save,
                verbose=verbose
            )[0].boxes.xyxy.detach().cpu().numpy()
            
            if len(yolo_bbox) < 1:
                print("No person detected")
                return None, None
            
            # Use the first (largest) detected bbox
            bbox_xyxy = yolo_bbox[0]
            
            # Convert to xywh format
            bbox_xywh = np.zeros(4)
            bbox_xywh[0] = bbox_xyxy[0]  # x
            bbox_xywh[1] = bbox_xyxy[1]  # y
            bbox_xywh[2] = abs(bbox_xyxy[2] - bbox_xyxy[0])  # width
            bbox_xywh[3] = abs(bbox_xyxy[3] - bbox_xyxy[1])  # height
            
            return bbox_xyxy, bbox_xywh
            
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return None, None


class SMPLestXManager:    
    def __init__(self, cfg, smplx_model_path):
        self.cfg = cfg
        self.smplx_model_path = smplx_model_path
        self.tester = None
        self.smpl_x = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            # Initialize SMPLest-X tester
            self.tester = Tester(self.cfg)
            print(f"Using 1 GPU.")
            print(f'Inference on images with [{self.cfg.model.pretrained_model_path}].')
            self.tester._make_model()
            
            # Initialize SMPL-X model
            self.smpl_x = SMPLX(self.smplx_model_path)
            print(f"SMPL-X model initialized from: {self.smplx_model_path}")
            
        except Exception as e:
            print(f"Error initializing SMPLest-X models: {e}")
            raise
    
    def predict(self, img_tensor, targets=None, meta_info=None):
        if self.tester is None:
            raise RuntimeError("SMPLest-X model not initialized")
        
        if targets is None:
            targets = {}
        if meta_info is None:
            meta_info = {}
        
        inputs = {'img': img_tensor}
        
        try:
            with torch.no_grad():
                out = self.tester.model(inputs, targets, meta_info, 'test')
            return out
        except Exception as e:
            print(f"Error during SMPLest-X prediction: {e}")
            raise
    
    def get_smplx_model(self):
        return self.smpl_x


def load_facial_landmarks(landmarks_dir, img_name):
    if not landmarks_dir:
        return None

    landmarks_file = os.path.join(landmarks_dir, img_name, f"landmarks.npy")
    if not os.path.exists(landmarks_file):
        print(f"Warning: Landmarks file not found: {landmarks_file}")
        return None
    
    return np.load(landmarks_file, allow_pickle=True).item()["ldm68"]

def compute_smplx_facial_landmarks(smplx_vertices, asset_paths, device='cuda'):
    # Load FLAME correspondence and embedding data
    embed_data = np.load(asset_paths['flame_embed_68'], allow_pickle=True)
    lmk_face_idx = torch.from_numpy(embed_data['lmk_face_idx']).long().to(device)  # (68,)
    lmk_b_coords = torch.from_numpy(embed_data['lmk_b_coords']).float().to(device)  # (68, 3)
    
    smplx_flame_vertex_ids = torch.from_numpy(np.load(asset_paths['flame_smplx_corr'], allow_pickle=True)).long().to(device)
    
    flame_data = pickle.load(open(asset_paths['flame_generic_model'], 'rb'), encoding='latin1')
    flame_faces = torch.from_numpy(flame_data['f']).long().to(device)  # (F, 3)
    
    landmarks = torch.zeros((68, 3), device=device)
    
    for i in range(68):
        # Get FLAME face (triangle) index and barycentric coords
        face_id = lmk_face_idx[i]
        b_coords = lmk_b_coords[i]  # [b0, b1, b2]
        
        # Get the 3 FLAME vertex IDs for this triangle
        flame_vertex_ids = flame_faces[face_id]  # [v0, v1, v2]
        
        # Map to SMPL-X vertex IDs
        smplx_vertex_ids = smplx_flame_vertex_ids[flame_vertex_ids]  # [smplx_v0, smplx_v1, smplx_v2]
        
        # Get the corresponding SMPL-X vertices
        tri_vertices = smplx_vertices[smplx_vertex_ids]  # Shape: (3, 3)
        
        # Compute landmark as weighted sum
        landmarks[i] = torch.sum(b_coords.unsqueeze(1) * tri_vertices, dim=0)  # b0*v0 + b1*v1 + b2*v2
    
    return landmarks


def project_3d_to_2d(points_3d, focal, princpt):
    fx, fy= focal
    cx, cy = princpt
    projected = torch.zeros((points_3d.shape[0], 2), device=points_3d.device)
    projected[:, 0] = points_3d[:, 0] * fx / points_3d[:, 2] + cx
    projected[:, 1] = points_3d[:, 1] * fy / points_3d[:, 2] + cy
    return projected

def optimize_smplx_landmarks(initial_params, gt_landmarks_2d, focal, princpt,
                           smplx_model, asset_paths, num_steps=50, lr=3e-3, landmark_weight=1.0, device='cuda'):
    optimizable_params = {}
    fixed_params = {}

    optimizable_params['smplx_expr'] = torch.tensor(initial_params['smplx_expr'], device=device, requires_grad=True)
    optimizable_params["smplx_jaw_pose"] = torch.tensor(initial_params['smplx_jaw_pose'], device=device, requires_grad=True)
    optimizable_params['cam_trans'] = torch.tensor(initial_params['cam_trans'], device=device, requires_grad=True)
    optimizable_params['smplx_shape'] = torch.tensor(initial_params['smplx_shape'], device=device, requires_grad=True)
    fixed_params['smplx_root_pose'] = torch.tensor(initial_params['smplx_root_pose'], device=device)
    fixed_params['smplx_body_pose'] = torch.tensor(initial_params['smplx_body_pose'], device=device)
    fixed_params['smplx_lhand_pose'] = torch.tensor(initial_params['smplx_lhand_pose'], device=device)
    fixed_params['smplx_rhand_pose'] = torch.tensor(initial_params['smplx_rhand_pose'], device=device)
    fixed_params['cam_trans'] = torch.tensor(initial_params['cam_trans'], device=device)

    # Setup optimizer - only optimize expression parameters
    # Setup optimizer with different learning rates for different parameter groups
    param_groups = [
        {'params': [optimizable_params['smplx_expr'], optimizable_params['smplx_jaw_pose'], optimizable_params['smplx_shape']], 'lr': lr},
        {'params': [optimizable_params['cam_trans']], 'lr': lr * 0.5}
    ]
    optimizer = torch.optim.AdamW(param_groups)
    
    # Convert GT landmarks to torch tensor
    gt_landmarks_2d = torch.tensor(gt_landmarks_2d, device=device, dtype=torch.float32)
    
    loss_history = []
    
    print(f"Starting facial expression optimization for {num_steps} steps...")
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass through SMPL-X
        cam_trans = fixed_params['cam_trans'] if step <= int(num_steps/2) else optimizable_params['cam_trans']
        smplx_output = smplx_model(
            global_orient=fixed_params['smplx_root_pose'].unsqueeze(0),
            body_pose=fixed_params['smplx_body_pose'].unsqueeze(0),
            left_hand_pose=fixed_params['smplx_lhand_pose'].unsqueeze(0),
            right_hand_pose=fixed_params['smplx_rhand_pose'].unsqueeze(0),
            jaw_pose=optimizable_params['smplx_jaw_pose'].unsqueeze(0),
            betas=optimizable_params['smplx_shape'].unsqueeze(0),
            expression=optimizable_params['smplx_expr'].unsqueeze(0),
            transl=cam_trans.unsqueeze(0),
            return_verts=True
        )
        
        vertices = smplx_output.vertices[0]  # Shape: (10475, 3)
        
        # Compute 3D facial landmarks
        landmarks_3d = compute_smplx_facial_landmarks(vertices, asset_paths, device)
        
        # Project to 2D
        landmarks_2d_pred = project_3d_to_2d(landmarks_3d, focal, princpt)

        # Compute loss
        landmark_loss = torch.nn.functional.l1_loss(landmarks_2d_pred, gt_landmarks_2d)
        total_loss = landmark_weight * landmark_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if step % 10 == 0:
            print(f"Step {step:3d}: Loss = {landmark_loss.item():.6f}")

    # Return optimized parameters - only expression is updated
    optimized_params = initial_params.copy()
    for key in optimizable_params:
        optimized_params[key] = optimizable_params[key].detach().cpu().numpy()
    return optimized_params, loss_history


def main(
    data_dir,
    output_dir,
    lmk_dir,
    ckpt_name='smplest_x_h',
    bust_assets_dir='/localhome/aha220/Hairdar/assets/bust/',
    use_yolo=True, num_optimization_steps=10, lr=8e-3, landmark_weight=1.0,
):
    
    cudnn.benchmark = True

    input_dir = data_dir
    facial_landmarks_dir = lmk_dir

    # Construct asset paths from bust_assets_dir
    asset_paths = get_bust_asset_paths(bust_assets_dir)
    
    # Validate that all required assets exist
    try:
        validate_asset_paths(asset_paths)
        print(f"All bust assets found in: {bust_assets_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    # init config
    config_path = os.path.join('modules/SMPLestX/pretrained_models', ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = os.path.join('modules/SMPLestX/pretrained_models', ckpt_name, f'{ckpt_name}.pth.tar')

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        }
    }
    cfg.update_config(new_config)

    # Initialize model managers
    yolo_manager = YOLOManager(use_yolo=use_yolo, cfg=cfg)
    smplestx_manager = SMPLestXManager(cfg=cfg, smplx_model_path="assets/body_models/base_models/smplx/parametric_models/lh/SMPLX_NEUTRAL.npz")
    
    # Get SMPL-X model for rendering
    smpl_x = smplestx_manager.get_smplx_model()

    print(f'Inference on images from [{input_dir}] with [{cfg.model.pretrained_model_path}].')

    # Find all image files in the input directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")

    processed_samples = []
    for img_path in tqdm(image_files, desc="Processing images"):
        
        # Get image name without extension for output folder creation
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Create output folder for this specific image
        image_output_folder = os.path.join(output_dir, img_name)
        os.makedirs(image_output_folder, exist_ok=True)
        
        print(f"Processing image: {img_name}")
        
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection using YOLO manager
        yolo_bbox_single, yolo_bbox_xywh = yolo_manager.detect_person(original_img)
        
        if yolo_bbox_single is None:
            # No person detected
            print(f"No person detected in {os.path.basename(img_path)}, saving original image")
            # Still save the original image for reference
            img_name_file = os.path.basename(img_path)
            output_img_name = f"no_detection_{img_name_file}"
            cv2.imwrite(os.path.join(image_output_folder, output_img_name), original_img[:, :, ::-1])
            continue
        # Process the single bbox
        bbox = process_bbox(bbox=yolo_bbox_xywh, 
                            img_width=original_img_width, 
                            img_height=original_img_height, 
                            input_img_shape=cfg.model.input_img_shape, 
                            ratio=getattr(cfg.data, "bbox_ratio", 1.0))                
        img, _, _ = generate_patch_image(cvimg=original_img, 
                                            bbox=bbox, 
                                            scale=1.0, 
                                            rot=0.0, 
                                            do_flip=False, 
                                            out_shape=cfg.model.input_img_shape)
            
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        out = smplestx_manager.predict(img, targets, meta_info)

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

        # render mesh
        focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                    cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
        princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                    cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
        
        # Extract SMPL-X parameters and camera parameters
        smplx_params = {
            'smplx_root_pose': out['smplx_root_pose'].detach().cpu().numpy()[0],  # [3]
            'smplx_body_pose': out['smplx_body_pose'].detach().cpu().numpy()[0],  # [63] 
            'smplx_lhand_pose': out['smplx_lhand_pose'].detach().cpu().numpy()[0],  # [45]
            'smplx_rhand_pose': out['smplx_rhand_pose'].detach().cpu().numpy()[0],  # [45]
            'smplx_jaw_pose': out['smplx_jaw_pose'].detach().cpu().numpy()[0],  # [3]
            'smplx_shape': out['smplx_shape'].detach().cpu().numpy()[0],  # [10]
            'smplx_expr': out['smplx_expr'].detach().cpu().numpy()[0],  # [10]
            'cam_trans': out['cam_trans'].detach().cpu().numpy()[0],  # [3]
            'smplx_joint_proj': out['smplx_joint_proj'].detach().cpu().numpy()[0],  # 2D joints
            'smplx_joint_cam': out['smplx_joint_cam'].detach().cpu().numpy()[0],  # 3D joints
            'smplx_mesh_cam': mesh,  # 3D mesh vertices
        }
        

        # Optimization with facial landmarks if requested
        if num_optimization_steps > 0 and facial_landmarks_dir:
            vis_img = original_img.copy()
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox_single[0]), int(yolo_bbox_single[1])), 
                                    (int(yolo_bbox_single[2]), int(yolo_bbox_single[3])), (0, 255, 0), 1)
            # draw mesh
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)

            # Save the rendered image before optimization
            pre_opt_img_name = f"pre_optimization_overlayed.png"
            cv2.imwrite(os.path.join(image_output_folder, pre_opt_img_name), vis_img[:, :, ::-1])

            # Load facial landmarks for this image
            img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            gt_landmarks_2d = load_facial_landmarks(facial_landmarks_dir, img_name_no_ext)

            if gt_landmarks_2d is not None:
                print(f"Loaded facial landmarks for {img_name_no_ext}, shape: {gt_landmarks_2d.shape}")

                # Create SMPL-X model for optimization
                smplx_model = smplx.create(
                    "assets/body_models/base_models/smplx/parametric_models/lh/SMPLX_NEUTRAL.npz",
                    model_type='smplx',
                    gender='neutral',
                    use_face_contour=True,
                    use_pca=False,
                    # flat_hand_mean=True,
                    batch_size=1
                ).cuda()

                # Optimize the parameters
                optimized_params, loss_history = optimize_smplx_landmarks(
                    initial_params=smplx_params,
                    gt_landmarks_2d=gt_landmarks_2d,
                    focal=focal,
                    princpt=princpt,
                    smplx_model=smplx_model,
                    asset_paths=asset_paths,
                    num_steps=num_optimization_steps,
                    lr=lr,
                    landmark_weight=landmark_weight,
                    device='cuda'
                )

                # Update parameters with optimized values
                smplx_params.update(optimized_params)

                # Recompute the mesh with optimized parameters
                with torch.no_grad():
                    optimized_smplx_output = smplx_model(
                        global_orient=torch.tensor(smplx_params['smplx_root_pose'], device='cuda').unsqueeze(0),
                        body_pose=torch.tensor(smplx_params['smplx_body_pose'], device='cuda').unsqueeze(0),
                        left_hand_pose=torch.tensor(smplx_params['smplx_lhand_pose'], device='cuda').unsqueeze(0),
                        right_hand_pose=torch.tensor(smplx_params['smplx_rhand_pose'], device='cuda').unsqueeze(0),
                        jaw_pose=torch.tensor(smplx_params['smplx_jaw_pose'], device='cuda').unsqueeze(0),
                        betas=torch.tensor(smplx_params['smplx_shape'], device='cuda').unsqueeze(0),
                        expression=torch.tensor(smplx_params['smplx_expr'], device='cuda').unsqueeze(0),
                        transl=torch.tensor(smplx_params['cam_trans'], device='cuda').unsqueeze(0),
                        return_verts=True
                    )
                    
                    # Update mesh with optimized vertices
                    mesh = optimized_smplx_output.vertices[0].detach().cpu().numpy()
                    smplx_params['smplx_mesh_cam'] = mesh
                
                print(f"Optimization completed. Final loss: {loss_history[-1]:.6f}")

            else:
                print(f"No facial landmarks found for {img_name_no_ext}, skipping optimization")
        elif num_optimization_steps > 0:
            print("Optimization requested but no facial_landmarks_dir provided")

        mesh_output_path = Path(image_output_folder) / "smplx.obj"
        try:
            save_smplx_mesh(mesh, smpl_x.face, mesh_output_path)
        except Exception as exc:
            print(f"Warning: Failed to save SMPL-X mesh for {img_name}: {exc}")
        
        # Camera parameters (computed focal and principal point)
        camera_params = {
            'focal': focal,  # [fx, fy]  
            'princpt': princpt,  # [cx, cy]
            'bbox': bbox,  # bounding box [x, y, w, h]
            'original_img_size': [original_img_width, original_img_height],
            'input_img_shape': cfg.model.input_img_shape
        }
        
        # Combine all parameters
        all_params = {
            'smplx_params': smplx_params,
            'camera_params': camera_params,
            'image_info': {
                'image_name': os.path.basename(img_path),
                'image_path': img_path,
                'yolo_bbox': yolo_bbox_single.tolist()
            }
        }

        vis_img = original_img.copy()
        # draw the bbox on img
        vis_img = cv2.rectangle(vis_img, (int(yolo_bbox_single[0]), int(yolo_bbox_single[1])), 
                                (int(yolo_bbox_single[2]), int(yolo_bbox_single[3])), (0, 255, 0), 1)
        # draw mesh
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)
        
        # Save parameters to files
        img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

        # Save as pickle file (most comprehensive, preserves numpy arrays)
        # param_filename = "params.pkl"
        # with open(os.path.join(image_output_folder, param_filename), 'wb') as f:
        #     pickle.dump(all_params, f)
        
        # Save as JSON file (human readable but limited precision)
        json_params = {
            'smplx_params': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                            for k, v in smplx_params.items()},
            'camera_params': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                for k, v in camera_params.items()},
            'image_info': all_params['image_info']
        }
        json_filename = "params.json"
        with open(os.path.join(image_output_folder, json_filename), 'w') as f:
            json.dump(json_params, f, indent=2)
        
        # Create separate parameter output directory
        # param_output_folder = os.path.join(image_output_folder, 'parameters')
        # os.makedirs(param_output_folder, exist_ok=True)
        
        # Save individual numpy arrays for structured access
        # for param_name, param_value in smplx_params.items():
        #     if isinstance(param_value, np.ndarray):
        #         np.save(os.path.join(param_output_folder, f"{param_name}.npy"), param_value)
        
        # Save camera parameters as numpy
        # np.save(os.path.join(param_output_folder, "focal.npy"), np.array(focal))
        # np.save(os.path.join(param_output_folder, "princpt.npy"), np.array(princpt))
        # np.save(os.path.join(param_output_folder, "bbox.npy"), np.array(bbox))

        # print(f"Parameters saved for image {img_name_no_ext}")

        # save rendered image
        img_name_final = os.path.basename(img_path)
        output_img_name = "overlayed.png"
        cv2.imwrite(os.path.join(image_output_folder, output_img_name), vis_img[:, :, ::-1])
        
        # Return paths to the saved parameter files
        processed_samples.append({
            'overlayed_image_path': os.path.join(image_output_folder, output_img_name),
            'pre_optimization_image_path': os.path.join(image_output_folder, pre_opt_img_name),
            'smplx_params_path': os.path.join(image_output_folder, json_filename),
            'smplx_mesh_path': str(mesh_output_path),
        })

    return processed_samples
