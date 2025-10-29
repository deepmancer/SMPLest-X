import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import trimesh

# PyTorch3D imports for modern rendering
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        HardPhongShader,
        PointLights,
        TexturesVertex,
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available. Falling back to basic vertex rendering.")

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    # Note: cfg not available in this context, these lines are unused
    # x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    # y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    # z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n') 
    obj_file.close()

def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices

def render_segmentation_mask(vertices, faces, cam_param, img_size):
    """
    Render a binary segmentation mask of the SMPL-X mesh using PyTorch3D.
    
    Args:
        vertices: Mesh vertices (N, 3) as numpy array
        faces: Mesh faces (M, 3) as numpy array
        cam_param: Camera parameters dict with 'focal' and 'princpt'
        img_size: Tuple of (height, width) for output mask
    
    Returns:
        Binary segmentation mask (H, W) as uint8 where 255=body, 0=background
    """
    if not PYTORCH3D_AVAILABLE:
        print("Warning: PyTorch3D not available. Cannot render segmentation mask.")
        return np.zeros(img_size, dtype=np.uint8)
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    img_h, img_w = img_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to torch tensors
    verts = torch.from_numpy(vertices).float().unsqueeze(0).to(device)  # (1, N, 3)
    faces_tensor = torch.from_numpy(faces).long().unsqueeze(0).to(device)  # (1, M, 3)
    
    # Create uniform white color for segmentation
    verts_rgb = torch.ones_like(verts)  # (1, N, 3) - all white
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # Create mesh
    mesh = Meshes(verts=verts, faces=faces_tensor, textures=textures)
    
    # Setup camera
    cameras = FoVPerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
        znear=0.01,
        zfar=100.0,
        fov=2 * np.arctan(img_h / (2 * focal[1])) * 180 / np.pi,
        aspect_ratio=img_w / img_h,
    )
    
    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    
    # No lighting needed for segmentation - use hard shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras)
    )
    
    # Render
    with torch.no_grad():
        images = renderer(mesh)  # (1, H, W, 4) RGBA
    
    # Extract alpha channel and create binary mask
    rendered_alpha = images[0, ..., 3].cpu().numpy()  # (H, W)
    mask = (rendered_alpha > 0).astype(np.uint8) * 255
    
    # Fix coordinate system: PyTorch3D renders upside down, and we need to flip horizontally
    # to match the original image coordinate system
    mask = cv2.flip(mask, -1)  # -1 flips both vertically and horizontally
    
    return mask


def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
    """
    Render a mesh onto an image using PyTorch3D.
    
    Args:
        img: Background image (H, W, 3) in RGB format
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
        cam_param: Camera parameters dict with 'focal' and 'princpt'
        mesh_as_vertices: If True, fall back to simple vertex projection
    
    Returns:
        Rendered image with mesh overlay (H, W, 3) as uint8
    """
    if mesh_as_vertices or not PYTORCH3D_AVAILABLE:
        # Fallback: simple vertex projection
        vertices_2d = perspective_projection(vertices.copy(), cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
        return img
    
    # PyTorch3D rendering
    focal, princpt = cam_param['focal'], cam_param['princpt']
    img_h, img_w = img.shape[:2]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to torch tensors
    verts = torch.from_numpy(vertices).float().unsqueeze(0).to(device)  # (1, N, 3)
    faces_tensor = torch.from_numpy(faces).long().unsqueeze(0).to(device)  # (1, M, 3)
    
    # Create a soft pink/peach color for the body (similar to original pyrender material)
    verts_rgb = torch.ones_like(verts) * torch.tensor([0.7, 0.7, 0.7], device=device)  # (1, N, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # Create mesh
    mesh = Meshes(verts=verts, faces=faces_tensor, textures=textures)
    
    # Setup camera - convert from intrinsic parameters to PyTorch3D format
    # PyTorch3D uses NDC coordinates, but we can use screen space with proper setup
    focal_length = torch.tensor([[focal[0], focal[1]]], device=device, dtype=torch.float32)
    principal_point = torch.tensor([[princpt[0], princpt[1]]], device=device, dtype=torch.float32)
    
    # Create perspective camera
    cameras = FoVPerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device).unsqueeze(0),  # Identity rotation
        T=torch.zeros(1, 3, device=device),  # No translation
        znear=0.01,
        zfar=100.0,
        fov=2 * np.arctan(img_h / (2 * focal[1])) * 180 / np.pi,  # Vertical FOV
        aspect_ratio=img_w / img_h,
    )
    
    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    
    # Setup lights (similar to pyrender's DirectionalLight with ambient)
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 3.0]],
        ambient_color=((0.3, 0.3, 0.3),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.1, 0.1, 0.1),),
    )
    
    # Create renderer with Phong shading
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    
    # Render
    with torch.no_grad():
        images = renderer(mesh)  # (1, H, W, 4) RGBA
    
    # Extract RGB and alpha
    rendered = images[0].cpu().numpy()  # (H, W, 4)
    rendered_rgb = rendered[..., :3]  # (H, W, 3)
    rendered_alpha = rendered[..., 3:4]  # (H, W, 1)
    
    # Fix coordinate system: PyTorch3D renders upside down and mirrored
    # Flip both vertically and horizontally to match OpenCV coordinate system
    rendered_rgb = cv2.flip(rendered_rgb, -1)  # -1 flips both axes
    rendered_alpha = cv2.flip(rendered_alpha, -1)
    
    # Blend with background
    alpha = 0.8  # Transparency factor
    valid_mask = (rendered_alpha > 0) * alpha
    
    img_float = img.astype(np.float32) / 255.0
    output_img = rendered_rgb * valid_mask + img_float * (1 - valid_mask)
    
    # Convert back to uint8
    img = (output_img * 255).clip(0, 255).astype(np.uint8)
    return img