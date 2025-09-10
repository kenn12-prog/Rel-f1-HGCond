# Step2: Generate node embeddings using pre-trained model

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from model import Model
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-dnf")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

# Load dataset and task
print("Loading dataset and task...")
dataset = get_dataset(args.dataset, download=True)
task = get_task(args.dataset, args.task, download=True)
out_channels = 1

# Build heterogeneous graph
print("Building graph...")
stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

print("Loading pre-trained model from Step 1...")
model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
).to(device)

# Load pre-trained model from Step1
model_path = f"./saved_models/{args.dataset}_{args.task}_step1.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run Step 1 first.")

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval() 
print("Model loaded successfully.")

# Create data loader for embedding generation
print("Creating data loader...")
train_table = task.get_table("train")
table_input = get_node_train_table_input(table=train_table, task=task)

# Use NeighborLoader for batch processing
loader = NeighborLoader(
    data,
    num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
    time_attr="time",
    input_nodes=table_input.nodes,
    input_time=table_input.time,
    batch_size=args.batch_size,
    temporal_strategy=args.temporal_strategy,
    shuffle=False,  # Keep consistent order for embedding generation
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
)

print(f"Created loader with {len(loader)} batches")

# Generate node embeddings
@torch.no_grad()
def generate_embeddings() -> Dict[str, torch.Tensor]:
    """
    Generate embeddings for all nodes using:
    1. Feature encoder
    2. Temporal encoder  
    3. Combine both encodings
    """
    all_embeddings = {}
    for node_type in data.node_types:
        all_embeddings[node_type] = []
    
    # Track node indices for each node type
    node_indices = {}
    for node_type in data.node_types:
        node_indices[node_type] = []

    all_batches = []
    for batch in loader:
        batch = batch.to(device)
        
        # Apply feature encoder
        feat_emb = model.encoder(batch.tf_dict)

        # Apply temporal encoder
        seed_time = batch[task.entity_table].seed_time
        time_emb = model.temporal_encoder(
            seed_time, 
            batch.time_dict, 
            batch.batch_dict
        )

        batch_embeddings = {}

        # Combine feature and temporal encodings
        for node_type in data.node_types:
            if node_type in feat_emb:
                # Get corresponding node IDs (n_id from NeighborLoader)
                node_ids = batch[node_type].n_id
                
                if node_type in time_emb:
                    # Nodes with temporal encoding: feature + time
                    final_embedding = feat_emb[node_type] + time_emb[node_type]
                else:
                    # Nodes without temporal encoding: feature only
                    final_embedding = feat_emb[node_type]
                
                # Store embeddings with corresponding node IDs
                batch_embeddings[node_type] = {
                    'embeddings': final_embedding.cpu(),
                    'node_indices': node_ids.cpu()
                }

        batch_embeddings['edge_index_dict'] = {
            k: v.cpu() for k, v in batch.edge_index_dict.items()
        }       
        all_batches.append(batch_embeddings)
    
    embedding_save_dir = Path("./saved_embeddings")
    embedding_save_dir.mkdir(parents=True, exist_ok=True)
    embedding_save_path = embedding_save_dir / f"{args.dataset}_{args.task}_embeddings_step2.pth"

    torch.save(all_batches, embedding_save_path)

if __name__ == "__main__":
    print("Starting embedding generation...")
    generate_embeddings()