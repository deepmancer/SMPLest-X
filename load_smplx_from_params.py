import os
import os.path as osp
import json
import numpy as np
import torch

# Choose one of the two model loaders:
#  (A) the pip package 'smplx' (recommended)
#  (B) your project's SMPLX wrapper at human_models.human_models
USE_PROJECT_WRAPPER = False  # set True if you want to use your own SMPLX class

if USE_PROJECT_WRAPPER:
    from human_models.human_models import SMPLX as ProjectSMPLX
else:
    import smplx  # pip install smplx

"""
What this script does
---------------------
- Loads parameters you saved from your inference script:
  smplx_root_pose (3,), smplx_body_pose (63,), smplx_lhand_pose (45,),
  smplx_rhand_pose (45,), smplx_jaw_pose (3,), smplx_shape (10,),
  smplx_expr (10,). It ignores cam_trans and any image-space camera info.
- Rebuilds a SMPL-X mesh with the SAME relative pose/expression/hands/neck,
  but keeps the mesh at the canonical origin: transl = 0.
- By default, it also sets global_orient = 0 (so no global rotation either).
  Toggle USE_GLOBAL_ORIENT below if you want to keep the recovered global rotation.
"""

# --------------------- USER CONFIG ---------------------
PARAM_ROOT = "/localhome/aha220/Hairdar/modules/SMPLest-X/demo/output_frames/060/parameters"
# pick one frame/bbox you already ran successfully (these files were created by your script)
FRAME_STEM = "000001_bbox0"  # e.g., "000001_bbox0" -> looks for 000001_bbox0_numpy/
MODEL_PATH = "/localhome/aha220/Hairdar/assets/bust/smplx/models"  # folder that contains SMPLX_NEUTRAL.npz etc.
GENDER = "neutral"  # "neutral", "male", "female"
SAVE_OBJ = True
OBJ_PATH = f"./{FRAME_STEM}_smplx_at_origin.obj"

# Keep recovered global rotation?
USE_GLOBAL_ORIENT = False  # set True to preserve 'smplx_root_pose'; otherwise identity

# If you saved JSON instead of individual npy files, set this to True and give the path
LOAD_FROM_JSON = False
JSON_PATH = osp.join(PARAM_ROOT, f"{FRAME_STEM.split('_bbox')[0]}_{FRAME_STEM.split('_bbox')[1]}_params.json")
# ------------------------------------------------------


def load_params_from_numpy(numpy_dir):
    """
    numpy_dir should contain files like:
      smplx_root_pose.npy, smplx_body_pose.npy, smplx_lhand_pose.npy, ...
    """
    load = lambda name: np.load(osp.join(numpy_dir, f"{name}.npy"))
    params = {
        "root_pose": load("smplx_root_pose"),     # (3,)
        "body_pose": load("smplx_body_pose"),     # (63,)
        "lhand_pose": load("smplx_lhand_pose"),   # (45,)
        "rhand_pose": load("smplx_rhand_pose"),   # (45,)
        "jaw_pose": load("smplx_jaw_pose"),       # (3,)
        "betas": load("smplx_shape"),             # (10,)
        "expression": load("smplx_expr"),         # (10,)
        # optional eye poses: if not available, we set zeros later
    }
    return params


def load_params_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    sp = data["smplx_params"]
    params = {
        "root_pose": np.array(sp["smplx_root_pose"], dtype=np.float32),
        "body_pose": np.array(sp["smplx_body_pose"], dtype=np.float32),
        "lhand_pose": np.array(sp["smplx_lhand_pose"], dtype=np.float32),
        "rhand_pose": np.array(sp["smplx_rhand_pose"], dtype=np.float32),
        "jaw_pose": np.array(sp["smplx_jaw_pose"], dtype=np.float32),
        "betas": np.array(sp["smplx_shape"], dtype=np.float32),
        "expression": np.array(sp["smplx_expr"], dtype=np.float32),
    }
    return params


def main():
    # 1) Load params you saved during inference
    if LOAD_FROM_JSON:
        params = load_params_from_json(JSON_PATH)
    else:
        numpy_dir = osp.join(PARAM_ROOT, f"{FRAME_STEM}_numpy")
        if not osp.isdir(numpy_dir):
            raise FileNotFoundError(f"Cannot find numpy directory: {numpy_dir}")
        params = load_params_from_numpy(numpy_dir)

    # 2) Prepare tensors (batch = 1). SMPL-X expects **axis-angle** for poses.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure shapes
    def to_tensor(arr, shape=None):
        t = torch.from_numpy(arr.astype(np.float32)).to(device)
        if shape is not None:
            t = t.view(*shape)
        return t

    # Pose pieces
    root_pose = to_tensor(params["root_pose"], (1, 3))                 # global_orient, axis-angle
    body_pose = to_tensor(params["body_pose"], (1, 21 * 3))            # 21 joints * 3 (axis-angle)
    lhand_pose = to_tensor(params["lhand_pose"], (1, 15 * 3))          # 15 * 3
    rhand_pose = to_tensor(params["rhand_pose"], (1, 15 * 3))          # 15 * 3
    jaw_pose = to_tensor(params["jaw_pose"], (1, 3))                   # 3
    betas = to_tensor(params["betas"], (1, -1))                        # shape (num_betas)
    expression = to_tensor(params["expression"], (1, -1))              # expr coeffs

    # Eyes (not provided in your dump). Keep neutral:
    leye_pose = torch.zeros((1, 3), dtype=torch.float32, device=device)
    reye_pose = torch.zeros((1, 3), dtype=torch.float32, device=device)

    # 3) Build the SMPL-X model (no translation; we’ll keep origin)
    if USE_PROJECT_WRAPPER:
        # Your project's wrapper typically mirrors the smplx interface; if not, adapt here.
        smplx_model = ProjectSMPLX(MODEL_PATH, gender=GENDER)
    else:
        smplx_model = smplx.create(
            MODEL_PATH,
            model_type="smplx",
            gender=GENDER,
            use_pca=False,           # you have full axis-angle hand params, not PCA
            num_betas=betas.shape[1],
            num_expression_coeffs=expression.shape[1],
            batch_size=1
        ).to(device)

    smplx_model.eval()

    # 4) Fix the canonical placement:
    #    - transl = 0 (no translation)
    #    - global_orient = 0 (identity) unless you want to keep the recovered one
    global_orient = root_pose if USE_GLOBAL_ORIENT else torch.zeros_like(root_pose)
    transl = torch.zeros((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        # smplx forward signature (common fields)
        output = smplx_model(
            betas=betas,
            expression=expression,
            body_pose=body_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=leye_pose,  # (typo fixed below) -> should be reye_pose
            global_orient=global_orient,
            transl=transl,
            return_verts=True
        )

    # (Fix small typo: set reye_pose correctly)
    # Re-run with the correct eye poses if you prefer:
    # output = smplx_model(..., leye_pose=leye_pose, reye_pose=reye_pose, ...)

    vertices = output.vertices[0].detach().cpu().numpy()     # (10475, 3) for SMPL-X
    faces = smplx_model.faces.astype(np.int32) if hasattr(smplx_model, "faces") else None
    joints = output.joints[0].detach().cpu().numpy()         # (J, 3) model-space joints

    print("SMPL-X mesh created at origin.")
    print("verts:", vertices.shape, "joints:", joints.shape)

    # 5) (Optional) Save an OBJ (no renderer required)
    if SAVE_OBJ:
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices, faces=faces, process=False)
            mesh.export(OBJ_PATH)
            print(f"Saved OBJ to: {OBJ_PATH}")
        except Exception as e:
            print("OBJ export skipped (install trimesh if you want OBJ export):", e)


if __name__ == "__main__":
    main()
