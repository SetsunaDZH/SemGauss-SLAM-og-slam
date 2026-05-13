from copy import deepcopy
from pathlib import Path

from configs.replica.replica import config as base_config

# Use a separate output folder name so the object-graph experiment does not
# overwrite the original SemGauss-SLAM results.
config = deepcopy(base_config)

repo_root = Path(__file__).resolve().parents[2]

# Prefer local checkpoints/data when this repo is run outside the original machine.
local_model = repo_root / "pth_files" / "replica" / "dinov2_replica.pth"
if local_model.exists():
    config["model"]["pretrained_model_path"] = str(local_model)

local_replica = repo_root / "data" / "replica"
if local_replica.exists():
    config["data"]["basedir"] = str(local_replica)
    configured_sequence = config["data"].get("sequence", "")
    if not (local_replica / configured_sequence).exists():
        available_sequences = sorted([p.name for p in local_replica.iterdir() if p.is_dir()])
        if available_sequences:
            config["data"]["sequence"] = available_sequences[0]

config["run_name"] = f"{config['data']['sequence']}_{config['seed']}_og"

# Lightweight object-centric Gaussian scene graph layer.
#
# The first version does not introduce a heavy instance segmentation network.
# It converts the existing SemGauss semantic output into connected-component
# object candidates and fuses them with the 3D Gaussian map through projection,
# depth consistency, and multi-frame assignment probabilities.
config["object_graph"] = dict(
    enabled=True,
    max_objects=256,

    # 2D semantic connected-component proposal settings.
    min_component_area=120,
    ignore_class_ids=[0, 255],

    # Gaussian-object association settings.
    assign_threshold=0.55,
    min_gaussians_per_object=30,
    lambda_mask=1.0,
    lambda_depth=0.1,
    lambda_semantic=0.5,
    depth_sigma=0.05,
    forget=0.01,

    # Object matching.  This first implementation is intentionally conservative
    # and mainly relies on semantic-class consistency before geometry becomes stable.
    match_iou_threshold=0.10,
    match_class_bonus=0.25,

    # Weak object and graph regularization terms.  Keep them small initially.
    anchor_weight=1.0e-4,
    graph_weight=5.0e-5,
    relation_min_confidence=0.5,

    # Geometry relation thresholds for adjacent/support edges.
    adjacent_distance=0.05,
    support_vertical_gap=0.08,
    support_min_overlap=0.05,
    huber_delta=0.05,
)

# Memory-safe overrides for local single-GPU runs:
# - Disable object graph regularization (auxiliary module).
# - Disable BA (major memory peak).
# - Disable adding new Gaussians over time to keep memory bounded.
# Conservative object-graph profile (enabled with reduced memory footprint).
config["object_graph"]["enabled"] = True
config["object_graph"]["max_objects"] = 16
config["object_graph"]["min_component_area"] = 600
config["object_graph"]["assign_threshold"] = 0.8
config["object_graph"]["min_gaussians_per_object"] = 200
config["object_graph"]["anchor_weight"] = 5.0e-5
config["object_graph"]["graph_weight"] = 1.0e-5
config["BA_every"] = 10**9
config["BA"]["num_iters"] = 0
config["mapping"]["add_new_gaussians"] = False

# Low-resolution run profile (roughly half of 680x1200).
config["data"]["desired_image_height"] = 340
config["data"]["desired_image_width"] = 600
config["model"]["H"] = 340
config["model"]["W"] = 600
config["run_name"] = f"{config['data']['sequence']}_{config['seed']}_og_lowres_objgraph_cons"

# Resume from latest available low-res checkpoint for faster reruns.
config["load_checkpoint"] = True
config["checkpoint_time_idx"] = 899
