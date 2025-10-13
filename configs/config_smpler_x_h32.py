config = {
  "human_model_path": "assets/body_models",
  "data": {
    "use_cache": True,
    "data_dir": "./data",

    # datasets
    "dataset_list": [
      "Human36M", "MSCOCO", "MPII", "AGORA", "EHF", "SynBody", "GTA_Human2",
      "EgoBody_Egocentric", "EgoBody_Kinect", "UBody", "PW3D", "MuCo", "PROX"
    ],
    "trainset_3d": ["MSCOCO", "AGORA", "UBody"],
    "trainset_2d": ["PW3D", "MPII", "Human36M"],
    "trainset_humandata": [
      "BEDLAM", "SPEC", "GTA_Human2", "SynBody", "PoseTrack",
      "EgoBody_Egocentric", "PROX", "CrowdPose",
      "EgoBody_Kinect", "MPI_INF_3DHP", "RICH", "MuCo", "InstaVariety",
      "Behave", "UP3D", "ARCTIC", "OCHuman", "CHI3D",
      "RenBody_HiRes", "MTP", "HumanSC3D", "RenBody",
      "FIT3D", "Talkshow", "SSP3D", "LSPET"
    ],
    "testset": "EHF",

    # sampling / downsample intervals
    "BEDLAM_train_sample_interval": 5,
    "EgoBody_Kinect_train_sample_interval": 10,
    "UBody_train_sample_interval": 10,             # from generic train_sample_interval
    "MPI_INF_3DHP_train_sample_interval": 5,
    "InstaVariety_train_sample_interval": 10,
    "RenBody_HiRes_train_sample_interval": 5,
    "ARCTIC_train_sample_interval": 10,
    "FIT3D_train_sample_interval": 10,
    "Talkshow_train_sample_interval": 10,

    # data strategy
    "bbox_ratio": 1.2,
    "no_aug": False,
    "data_strategy": "balance",
    "total_data_len": 4500000,

    # dataset-specific toggles
    "agora_fix_betas": True,
    "agora_fix_global_orient_transl": True,
    "agora_valid_root_pose": True,

    # ubody / loader options
    "test_sample_interval": 100,
    "make_same_len": False,
  },

  "train": {
    "num_gpus": -1,
    "continue_train": False,
    "start_over": True,
    "end_epoch": 10,
    "train_batch_size": 16,
    "num_thread": 2,
    "lr": 1e-5,
    # optional extras from your setup
    "save_epoch": 1,
    "remove_checkpoint": False,
    "print_iters": 100,
    "syncbn": True,
    "lr_mult": 1,

    # loss weights
    "smplx_kps_3d_weight": 100.0,
    "smplx_kps_2d_weight": 1.0,
    "smplx_pose_weight": 10.0,
    "smplx_shape_weight": 1.0,   # mapped from smplx_loss_weight
    # the following were not in the source; only included if needed elsewhere
    # "smplx_orient_weight": 1.0,
    # "hand_root_weight": 1.0,
    # "hand_consist_weight": 1.0,
  },

  "inference": {
    "num_gpus": -1,
    "detection": {
      # not specified in the second config; fill in if you have a detector
      "model_type": None,
      "model_path": None,
      "conf": 0.5,
      "save": False,
      "verbose": False,
      "iou_thr": 0.5,
    },
  },

  "test": {
    "test_batch_size": 32,
    "vis": False
  },

  "model": {
    "model_type": "smpler_x_h",
    "agora_benchmark": "agora_model",  # 'agora_model' or 'test_only'

    # encoder / decoder
    "encoder_config_file": "transformer_utils/configs/smpler_x/encoder/body_encoder_huge.py",
    "encoder_pretrained_model_path": "../pretrained_models/vitpose_huge.pth",
    "decoder_config": {
      "feat_dim": 1280
    },

    # io shapes & camera
    "input_img_shape": (512, 384),
    "input_body_shape": (256, 192),
    "output_hm_shape": (16, 16, 12),
    "input_hand_shape": (256, 256),
    "output_hand_hm_shape": (16, 16, 16),
    "input_face_shape": (192, 192),
    "output_face_hm_shape": (8, 8, 8),

    "focal": (int(1024 * 50/36), int(1024 * 50/36)),
    "princpt": (192 / 2, 256 / 2),
    "body_3d_size": 2,
    "hand_3d_size": 0.3,
    "face_3d_size": 0.3,
    "camera_3d_size": 2.5,

    # fixed args
    "upscale": 4,
    "hand_pos_joint_num": 20,
    "face_pos_joint_num": 72,
    "num_task_token": 24,
    "num_noise_sample": 0,
  },

  "log": {
    "exp_name": "output/exp1/pre_analysis",
    "output_dir": None,
    "model_dir": None,
    "log_dir": None,
    "result_dir": None
  }
}
