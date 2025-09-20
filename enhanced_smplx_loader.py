#!/usr/bin/env python3
"""
Enhanced SMPL-X loader with proper coordinate handling and expression verification
"""

import numpy as np
import pickle
import torch
import smplx
import trimesh

def load_and_create_smplx_with_expressions(param_file, output_dir="./"):
    """
    Load SMPL-X parameters and create model with proper expressions
    """
    
    # Load saved parameters
    with open(param_file, 'rb') as f:
        params = pickle.load(f)
    
    # Initialize SMPL-X model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    smplx_model = smplx.create(
        model_path='/localhome/aha220/Hairdar/assets/bust/smplx/models/',
        model_type="smplx",
        gender="neutral",
        use_hands=True,
        use_face_contour=False,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=1,
        ext='npz'
    ).to(device)
    
    # Extract parameters
    smplx_params = params['smplx_params']
    
    # Convert to tensors
    def to_tensor(x):
        return torch.from_numpy(x).float().unsqueeze(0).to(device)
    
    print("SMPL-X Parameters:")
    print(f"  Shape (betas): {smplx_params['smplx_shape']}")
    print(f"  Expression: {smplx_params['smplx_expr']} (max: {np.abs(smplx_params['smplx_expr']).max():.3f})")
    print(f"  Root pose: {smplx_params['smplx_root_pose']}")
    print(f"  Jaw pose: {smplx_params['smplx_jaw_pose']}")
    
    # Create SMPL-X output with expressions
    output_with_expr = smplx_model(
        global_orient=to_tensor(smplx_params['smplx_root_pose']),
        body_pose=to_tensor(smplx_params['smplx_body_pose']),
        left_hand_pose=to_tensor(smplx_params['smplx_lhand_pose']),
        right_hand_pose=to_tensor(smplx_params['smplx_rhand_pose']),
        jaw_pose=to_tensor(smplx_params['smplx_jaw_pose']),
        betas=to_tensor(smplx_params['smplx_shape']),
        expression=to_tensor(smplx_params['smplx_expr']),
        transl=to_tensor(smplx_params['cam_trans']),  # Include translation
        return_verts=True
    )
    
    # Create SMPL-X output without expressions (for comparison)
    output_no_expr = smplx_model(
        global_orient=to_tensor(smplx_params['smplx_root_pose']),
        body_pose=to_tensor(smplx_params['smplx_body_pose']),
        left_hand_pose=to_tensor(smplx_params['smplx_lhand_pose']),
        right_hand_pose=to_tensor(smplx_params['smplx_rhand_pose']),
        jaw_pose=to_tensor(smplx_params['smplx_jaw_pose']),
        betas=to_tensor(smplx_params['smplx_shape']),
        expression=torch.zeros(1, 10).to(device),  # Zero expressions
        transl=to_tensor(smplx_params['cam_trans']),
        return_verts=True
    )
    
    # Get vertices and faces
    vertices_with_expr = output_with_expr.vertices.detach().cpu().numpy()[0]
    vertices_no_expr = output_no_expr.vertices.detach().cpu().numpy()[0]
    faces = smplx_model.faces
    
    # Calculate expression impact
    vertex_diff = np.linalg.norm(vertices_with_expr - vertices_no_expr, axis=1)
    
    print(f"\nMesh Information:")
    print(f"  Vertices: {vertices_with_expr.shape}")
    print(f"  Faces: {faces.shape}")
    print(f"  Joints: {output_with_expr.joints.shape}")
    
    print(f"\nExpression Impact Analysis:")
    print(f"  Max vertex displacement: {vertex_diff.max():.6f}")
    print(f"  Mean vertex displacement: {vertex_diff.mean():.6f}")
    print(f"  Vertices moved >0.5mm: {np.sum(vertex_diff > 0.0005)}")
    print(f"  Vertices moved >1.0mm: {np.sum(vertex_diff > 0.001)}")
    print(f"  Vertices moved >2.0mm: {np.sum(vertex_diff > 0.002)}")
    
    # Find most affected vertices (likely face region)
    most_affected_indices = np.argsort(vertex_diff)[-50:]  # Top 50 most affected
    print(f"  Most affected vertex displacement: {vertex_diff[most_affected_indices[-1]]:.6f}")
    
    # Save meshes
    mesh_with_expr = trimesh.Trimesh(vertices=vertices_with_expr, faces=faces)
    mesh_no_expr = trimesh.Trimesh(vertices=vertices_no_expr, faces=faces)
    
    expr_file = f"{output_dir}/smplx_with_expressions.obj"
    neutral_file = f"{output_dir}/smplx_neutral.obj"
    
    mesh_with_expr.export(expr_file)
    mesh_no_expr.export(neutral_file)
    
    print(f"\nSaved Files:")
    print(f"  With expressions: {expr_file}")
    print(f"  Neutral (no expr): {neutral_file}")
    
    # Create a displacement map for visualization
    displacement_colors = vertex_diff / vertex_diff.max()  # Normalize to [0,1]
    mesh_displacement = trimesh.Trimesh(vertices=vertices_with_expr, faces=faces)
    
    # Color vertices by displacement (red = more displacement)
    colors = np.zeros((len(vertices_with_expr), 4))
    colors[:, 0] = displacement_colors  # Red channel
    colors[:, 3] = 1.0  # Alpha
    mesh_displacement.visual.vertex_colors = colors
    
    displacement_file = f"{output_dir}/smplx_displacement_map.obj"
    mesh_displacement.export(displacement_file)
    print(f"  Displacement map: {displacement_file}")
    
    # Verify against original saved mesh if available
    if 'smplx_mesh_cam' in smplx_params:
        original_mesh = smplx_params['smplx_mesh_cam']
        reconstruction_error = np.abs(original_mesh - vertices_with_expr).mean()
        print(f"\nReconstruction vs Original:")
        print(f"  Mean error: {reconstruction_error:.6f}")
        
        if reconstruction_error > 1.0:
            print("  ⚠️  High reconstruction error - coordinate system may differ")
            
            # Try without translation
            output_no_trans = smplx_model(
                global_orient=to_tensor(smplx_params['smplx_root_pose']),
                body_pose=to_tensor(smplx_params['smplx_body_pose']),
                left_hand_pose=to_tensor(smplx_params['smplx_lhand_pose']),
                right_hand_pose=to_tensor(smplx_params['smplx_rhand_pose']),
                jaw_pose=to_tensor(smplx_params['smplx_jaw_pose']),
                betas=to_tensor(smplx_params['smplx_shape']),
                expression=to_tensor(smplx_params['smplx_expr']),
                # transl=torch.zeros(1, 3).to(device),  # No translation
                return_verts=True
            )
            vertices_no_trans = output_no_trans.vertices.detach().cpu().numpy()[0]
            error_no_trans = np.abs(original_mesh - vertices_no_trans).mean()
            print(f"  Error without translation: {error_no_trans:.6f}")
            
            if error_no_trans < reconstruction_error:
                print("  ✅ Better match without global translation")
                # Save the version without translation
                mesh_no_trans = trimesh.Trimesh(vertices=vertices_no_trans, faces=faces)
                best_file = f"{output_dir}/smplx_best_match.obj"
                mesh_no_trans.export(best_file)
                print(f"  Best match saved: {best_file}")
    
    return {
        'vertices_with_expr': vertices_with_expr,
        'vertices_no_expr': vertices_no_expr,
        'faces': faces,
        'expression_impact': vertex_diff,
        'smplx_output': output_with_expr
    }

if __name__ == "__main__":
    param_file = "/localhome/aha220/Hairdar/modules/SMPLest-X/demo/output_frames/060/parameters/000001_bbox0_params.pkl"
    result = load_and_create_smplx_with_expressions(param_file)
    
    print("\n✅ SMPL-X reconstruction complete with facial expressions!")
    print("Check the generated OBJ files to see the differences between:")
    print("  - smplx_with_expressions.obj (full model with expressions)")
    print("  - smplx_neutral.obj (same pose but no facial expressions)")
    print("  - smplx_displacement_map.obj (colored by expression displacement)")