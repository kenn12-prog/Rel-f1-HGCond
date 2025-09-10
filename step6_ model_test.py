# Step6: Hybrid model evaluation

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from model import Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

# Import CompressedHeteroGNN from step5
from step5 import CompressedHeteroGNN

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


class HybridModel(nn.Module):
    """
    Hybrid model: combines original dataset's encoder+temporal_encoder with compressed heterogeneous GNN
    
    Key insight: CompressedHeteroGNN was trained on encoder+temporal_encoder features, so can be used directly
    """
    def __init__(self, original_model: Model, compressed_hetero_gnn, 
                 target_node_type: str, feature_dim: int):
        super().__init__()
        
        # Extract encoder and temporal_encoder from original model
        self.encoder = original_model.encoder
        self.temporal_encoder = original_model.temporal_encoder
        
        # Use complete compressed heterogeneous graph GNN (core achievement from step5 training)
        self.compressed_hetero_gnn = compressed_hetero_gnn
        
        # Freeze encoder and temporal_encoder (use pre-trained weights)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
        
        # Compressed graph GNN can remain trainable or be frozen
        # Here we freeze it for inference only
        for param in self.compressed_hetero_gnn.parameters():
            param.requires_grad = False
        
        self.target_node_type = target_node_type
        self.feature_dim = feature_dim
    
    def forward(self, batch, entity_table):
        """
        Forward pass:
        1. Use original model's encoder and temporal_encoder to process input
        2. Directly use compressed graph GNN for prediction (features are compatible)
        """
        seed_time = batch[entity_table].seed_time
        
        # Step 1: Use original model's encoder to process features
        with torch.no_grad():  # encoder part doesn't need gradients
            x_dict = self.encoder(batch.tf_dict)
            
            # Step 2: Use original model's temporal_encoder to process time information
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            
            # Step 3: Combine features and temporal encodings (this is step2's output)
            for node_type, rel_time in rel_time_dict.items():
                if node_type in x_dict:
                    x_dict[node_type] = x_dict[node_type] + rel_time
        
        # Step 4: Extract target node features
        target_features = x_dict[entity_table][:seed_time.size(0)]
        
        # Step 5: Prepare features for compressed hetero GNN
        # We need to create a minimal x_dict and edge_index_dict for the compressed GNN
        # Since this is inference only, we'll create a minimal structure
        
        with torch.no_grad():  # compressed graph GNN is frozen
            # Create minimal x_dict with only target node type
            x_dict = {self.target_node_type: target_features}
            
            # Create empty edge_index_dict (no edges needed for inference in this case)
            edge_index_dict = {}
            
            # Use compressed hetero GNN for prediction
            try:
                pred = self.compressed_hetero_gnn(x_dict, edge_index_dict, self.target_node_type)
            except Exception as e:
                # Fallback: if compressed GNN fails, use simple linear projection
                print(f"Warning: Compressed GNN failed ({e}), using target features directly")
                if hasattr(self.compressed_hetero_gnn, 'head'):
                    pred = self.compressed_hetero_gnn.head(target_features)
                else:
                    # Ultimate fallback: return zeros
                    pred = torch.zeros(target_features.size(0), 1, device=target_features.device)
        
        return pred


def load_models():
    """
    Load Step1's original model and Step5's compressed graph GNN model
    """
    print("Loading models...")
    
    # 1. Load dataset information (needed for rebuilding original model)
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)
    
    # 2. Rebuild graph data (for original model structure)
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
    
    # 3. Determine task type and output dimension
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")
    
    # 4. Rebuild original model structure
    original_model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
        norm="batch_norm",
    ).to(device)
    
    # 5. Load Step1 model weights
    step1_model_path = f"./saved_models/{args.dataset}_{args.task}_step1.pth"
    if not os.path.exists(step1_model_path):
        raise FileNotFoundError(f"Step1 model not found at {step1_model_path}")
    
    step1_checkpoint = torch.load(step1_model_path, map_location=device)
    original_model.load_state_dict(step1_checkpoint)
    print(f"Loaded Step1 model from {step1_model_path}")
    
    # 6. Load Step5 compressed graph GNN model
    step5_model_path = f"./saved_models/compressed_gnn_{args.dataset}_{args.task}_best.pth"
    if not os.path.exists(step5_model_path):
        raise FileNotFoundError(f"Step5 compressed GNN model not found at {step5_model_path}")
    
    step5_checkpoint = torch.load(step5_model_path, map_location=device)
    step5_args = step5_checkpoint['args']
    
    # Rebuild compressed graph GNN model structure
    # First, load step4 compressed graph to get node/edge types
    # from step4 import load_compressed_graph as load_step4_compressed
    from step4_no_convolution import load_compressed_graph as load_step4_compressed
    
    compressed_data, _ = load_step4_compressed(
            args.dataset, args.task, getattr(step5_args, 'compression_rate', 0.1)
    )
        
        # Get valid node and edge types from compressed data
    valid_node_types = []
    for node_type in compressed_data.node_types:
            node_features = compressed_data[node_type].x
            if node_features is not None and node_features.size(0) > 0:
                valid_node_types.append(node_type)
        
    valid_edge_types = []
    for edge_type in compressed_data.edge_types:
            edge_index = compressed_data[edge_type].edge_index
            if edge_index is not None and edge_index.size(1) > 0:
                src_type, _, dst_type = edge_type
                src_features = compressed_data[src_type].x if src_type in compressed_data.node_types else None
                dst_features = compressed_data[dst_type].x if dst_type in compressed_data.node_types else None
                
                if (src_features is not None and src_features.size(0) > 0 and 
                    dst_features is not None and dst_features.size(0) > 0):
                    valid_edge_types.append(edge_type)
        
        # Get feature dimension from target node type
    target_node_type = task.entity_table
    if target_node_type in compressed_data.node_types:
            feature_dim = compressed_data[target_node_type].x.size(1)
    else:
            feature_dim = step5_args.channels
        
    compressed_gnn = CompressedHeteroGNN(
            node_types=valid_node_types,
            edge_types=valid_edge_types,
            input_dim=feature_dim,
            hidden_dim=step5_args.channels,
            output_dim=out_channels,
            aggr=step5_args.aggr,
            num_layers=step5_args.num_layers
    ).to(device)
    
    compressed_gnn.load_state_dict(step5_checkpoint['model_state_dict'])
    print(f"Loaded Step5 compressed GNN from {step5_model_path}")
    print(f"  Aggregation: {getattr(step5_args, 'aggr', 'sum')}")
    print(f"  Layers: {step5_args.num_layers}")
    print(f"  Node types: {len(compressed_gnn.node_types)}")
    print(f"  Edge types: {len(compressed_gnn.edge_types)}")
    
    return original_model, compressed_gnn, data, col_stats_dict, task


def create_hybrid_model(original_model, compressed_gnn, task):
    """
    Create hybrid model
    """
    hybrid_model = HybridModel(
        original_model=original_model,
        compressed_hetero_gnn=compressed_gnn,
        target_node_type=task.entity_table,
        feature_dim=args.channels
    ).to(device)
    
    print("Created hybrid model:")
    print(f"  Target node type: {task.entity_table}")
    print(f"  Feature dimension: {args.channels}")
    print(f"  Architecture: Original Encoder+Temporal → Complete Compressed GNN")
    print(f"  Using full step5 trained model: {type(compressed_gnn).__name__}")
    print(f"  All components frozen for inference")
    
    return hybrid_model


@torch.no_grad()
def test_hybrid_model(hybrid_model, loader: NeighborLoader, task, clamp_min=None, clamp_max=None) -> np.ndarray:
    """
    Test hybrid model
    """
    hybrid_model.eval()
    
    pred_list = []
    for batch in tqdm(loader, desc="Testing hybrid model"):
        batch = batch.to(device)
        pred = hybrid_model(batch, task.entity_table)
        
        # Handle clamp for regression tasks
        if task.task_type == TaskType.REGRESSION:
            if clamp_min is not None and clamp_max is not None:
                pred = torch.clamp(pred, clamp_min, clamp_max)
        
        # Handle sigmoid for classification tasks
        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)
        
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    
    pred_array = torch.cat(pred_list, dim=0).numpy()
    return pred_array


def main():
    """
    Main function: Step6 hybrid model evaluation
    """
    print("=== Step6: Hybrid Model Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    
    # 1. Load models
    original_model, compressed_gnn, data, col_stats_dict, task = load_models()
    
    # 2. Create hybrid model
    print("\n=== Creating Hybrid Model ===")
    hybrid_model = create_hybrid_model(original_model, compressed_gnn, task)
    
    # 3. Prepare test data loaders
    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.REGRESSION:
        # Get clamp values
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )
        print(f"Regression clamp range: [{clamp_min:.4f}, {clamp_max:.4f}]")
    
    # 4. Create data loaders
    loader_dict: Dict[str, NeighborLoader] = {}
    for split in ["val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    
    # 5. Validation set evaluation
    print("\n=== Validation Set Evaluation ===")
    val_pred = test_hybrid_model(hybrid_model, loader_dict["val"], task, clamp_min, clamp_max)
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    print(f"Hybrid Model Val metrics: {val_metrics}")
    
    # 6. Test set evaluation
    print("\n=== Test Set Evaluation ===")
    test_pred = test_hybrid_model(hybrid_model, loader_dict["test"], task, clamp_min, clamp_max)
    test_metrics = task.evaluate(test_pred)
    print(f"Hybrid Model Test metrics: {test_metrics}")
    
    # 7. Compare with original Step1 model
    print("\n=== Comparison with Step1 Original Model ===")
    original_model.eval()
    
    # Step1 original model performance on validation set
    @torch.no_grad()
    def test_original_model(model, loader):
        model.eval()
        pred_list = []
        for batch in tqdm(loader, desc="Testing original model"):
            batch = batch.to(device)
            pred = model(batch, task.entity_table)
            
            if task.task_type == TaskType.REGRESSION:
                if clamp_min is not None and clamp_max is not None:
                    pred = torch.clamp(pred, clamp_min, clamp_max)
            
            if task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                pred = torch.sigmoid(pred)
            
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()
    
    step1_val_pred = test_original_model(original_model, loader_dict["val"])
    step1_val_metrics = task.evaluate(step1_val_pred, task.get_table("val"))
    print(f"Step1 Original Model Val metrics: {step1_val_metrics}")
    
    step1_test_pred = test_original_model(original_model, loader_dict["test"])
    step1_test_metrics = task.evaluate(step1_test_pred)
    print(f"Step1 Original Model Test metrics: {step1_test_metrics}")
    
    # 8. Summary comparison
    print("\n=== Summary Comparison ===")
    print("Validation Set:")
    print(f"  Step1 Original: {step1_val_metrics}")
    print(f"  Step6 Hybrid:   {val_metrics}")
    
    print("\nTest Set:")
    print(f"  Step1 Original: {step1_test_metrics}")
    print(f"  Step6 Hybrid:   {test_metrics}")
    
    # Calculate performance differences
    print("\n=== Performance Comparison ===")
    for metric_name in step1_test_metrics.keys():
        if metric_name in test_metrics:
            step1_value = step1_test_metrics[metric_name]
            step6_value = test_metrics[metric_name]
            if isinstance(step1_value, (int, float)) and isinstance(step6_value, (int, float)):
                diff = step6_value - step1_value
                diff_pct = (diff / step1_value) * 100 if step1_value != 0 else 0
                print(f"{metric_name}: {step1_value:.4f} -> {step6_value:.4f} ({diff_pct:+.2f}%)")
    
    print("\nStep6 hybrid model evaluation completed!")


if __name__ == "__main__":
    main()
