# Step5: Training on compressed graph

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

# Import GraphCompressor from step4
from step4 import GraphCompressor

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-dnf")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--compression_rate", type=float, default=0.1)
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

class SafeHeteroGraphSAGE(nn.Module):
    """
    A safe version of HeteroGraphSAGE that handles missing node types gracefully
    """
    def __init__(self, node_types, edge_types, channels, aggr="sum", num_layers=2):
        super().__init__()
        
        from torch_geometric.nn import HeteroConv, SAGEConv
        from torch_geometric.nn.norm import LayerNorm
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        # Filter out None values at each layer
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Only pass valid x_dict and edge_index_dict to conv
            valid_x_dict = {k: v for k, v in x_dict.items() if v is not None}
            
            # Filter edge_index_dict to only include edges between valid nodes
            valid_edge_index_dict = {}
            for edge_type, edge_index in edge_index_dict.items():
                if edge_index is not None and edge_index.size(1) > 0:
                    src_type, _, dst_type = edge_type
                    if src_type in valid_x_dict and dst_type in valid_x_dict:
                        valid_edge_index_dict[edge_type] = edge_index
            
            if not valid_x_dict or not valid_edge_index_dict:
                break
                
            # Apply convolution
            out_dict = conv(valid_x_dict, valid_edge_index_dict)
            
            # Apply normalization and activation only to valid outputs
            x_dict = {}
            for node_type in self.node_types:
                if node_type in out_dict and out_dict[node_type] is not None:
                    x = norm_dict[node_type](out_dict[node_type])
                    x_dict[node_type] = x.relu()
                else:
                    x_dict[node_type] = None

        return x_dict


class CompressedHeteroGNN(nn.Module):
    """
    Heterogeneous GNN model for compressed graphs following RelBench approach
    Uses HeteroConv to handle all edge types automatically
    """
    def __init__(self, node_types: list, edge_types: list, input_dim: int, 
                 hidden_dim: int, output_dim: int, aggr: str = "sum", num_layers: int = 2):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        
        # Import RelBench's modules
        from torch_geometric.nn import MLP
        
        # Use our SafeHeteroGraphSAGE instead of RelBench's version
        self.gnn = SafeHeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            channels=hidden_dim,
            aggr=aggr,
            num_layers=num_layers,
        )
        
        # Node-specific input projections (only for valid node types)
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            self.input_projections[node_type] = nn.Linear(input_dim, hidden_dim)
        
        # Output head for target node type only
        self.head = MLP(
            hidden_dim,
            out_channels=output_dim,
            norm="batch_norm",
            num_layers=1,
            dropout=0.3
        )
        
    def reset_parameters(self):
        """Reset all parameters"""
        self.gnn.reset_parameters()
        for proj in self.input_projections.values():
            proj.reset_parameters()
        self.head.reset_parameters()
        
    def forward(self, x_dict, edge_index_dict, target_node_type: str, target_mask=None):
        """
        Forward pass for heterogeneous compressed graph
        
        Args:
            x_dict: Dict of node features {node_type: features}
            edge_index_dict: Dict of edge indices {edge_type: edge_index}
            target_node_type: The node type we want to predict
            target_mask: Mask for target nodes (if only predicting subset)
        """
        # Filter out None values and project all node types to same dimension
        projected_x_dict = {}
        for node_type, x in x_dict.items():
            if x is not None and x.size(0) > 0:
                projected_x_dict[node_type] = self.input_projections[node_type](x)
        
        # Filter out None/empty edge indices
        filtered_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index is not None and edge_index.size(1) > 0:
                # Also check if both source and target node types exist in projected_x_dict
                src_type, _, dst_type = edge_type
                if src_type in projected_x_dict and dst_type in projected_x_dict:
                    filtered_edge_index_dict[edge_type] = edge_index
        
        # Debug
        if len(filtered_edge_index_dict) == 0:
            print(f"Warning: No valid edges found for GNN processing")
            print(f"Available projected nodes: {list(projected_x_dict.keys())}")
            print(f"Available edges: {list(edge_index_dict.keys())}")
        
        # Apply heterogeneous GNN with filtered data
        out_dict = self.gnn(projected_x_dict, filtered_edge_index_dict)
        
        # Get target node embeddings (handle None case)
        if target_node_type not in out_dict or out_dict[target_node_type] is None:
            raise ValueError(f"Target node type {target_node_type} has no valid embeddings after GNN processing")
        
        target_x = out_dict[target_node_type]
        
        # Apply mask if provided
        if target_mask is not None:
            target_x = target_x[target_mask]
        
        # Final prediction
        return self.head(target_x)


def load_compressed_graph():
    """
    Load compressed graph from step4 results
    """
    print("Loading compressed graph from step4...")
    
    # Import load function from step4
    # from step4 import load_compressed_graph as load_step4_compressed
    from step4_no_convolution import load_compressed_graph as load_step4_compressed
    
    try:
        compressed_data, compressed_labels = load_step4_compressed(
            args.dataset, args.task, args.compression_rate
        )
        return compressed_data, compressed_labels
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please run step4 first to create compressed graph:")
        print(f"  python step4.py --dataset {args.dataset} --task {args.task} --compression_rate {args.compression_rate}")
        raise


def evaluate_model(model, compressed_data, target_node_type, labels, task_type, split_mask=None):
    """
    Evaluate heterogeneous compressed graph model performance
    """
    model.eval()
    with torch.no_grad():
        if target_node_type not in compressed_data.node_types:
            return {"error": f"Target node type {target_node_type} not found"}
        
        # Prepare x_dict for all node types (filter out None/empty nodes)
        x_dict = {}
        for node_type in compressed_data.node_types:
            node_features = compressed_data[node_type].x
            if node_features is not None and node_features.size(0) > 0:
                x_dict[node_type] = node_features.to(device)
        
        # Prepare edge_index_dict (filter out None/empty edges and check node validity)
        edge_index_dict = {}
        for edge_type in compressed_data.edge_types:
            edge_index = compressed_data[edge_type].edge_index
            if edge_index is not None and edge_index.size(1) > 0:
                # Check if both source and target node types exist in x_dict
                src_type, _, dst_type = edge_type
                if src_type in x_dict and dst_type in x_dict:
                    edge_index_dict[edge_type] = edge_index.to(device)
        
        # Get target labels
        y_true = labels.to(device)
        if split_mask is not None:
            y_true = y_true[split_mask]
        
        # Forward pass using heterogeneous model
        pred = model(x_dict, edge_index_dict, target_node_type, split_mask)
        
        # Calculate metrics
        if task_type == TaskType.BINARY_CLASSIFICATION:
            pred_prob = torch.sigmoid(pred).cpu().numpy().flatten()
            y_true_np = y_true.cpu().numpy().flatten()
            auc = roc_auc_score(y_true_np, pred_prob)
            pred_binary = (pred_prob > 0.5).astype(int)
            f1 = f1_score(y_true_np, pred_binary)
            return {"auc": auc, "f1": f1}
        
        elif task_type == TaskType.REGRESSION:
            pred_np = pred.cpu().numpy().flatten()
            y_true_np = y_true.cpu().numpy().flatten()
            mae = mean_absolute_error(y_true_np, pred_np)
            return {"mae": mae}
        
        else:
            return {"error": f"Unsupported task type: {task_type}"}


def train_compressed_model():
    """
    Main function for training compressed graph model
    """
    # 1. Load task and dataset information
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)
    
    # 2. Load compressed graph from step4
    compressed_data, compressed_labels = load_compressed_graph()
    
    print(f"Compressed graph loaded:")
    for node_type in compressed_data.node_types:
        compressed_count = compressed_data[node_type].num_nodes
        print(f"  {node_type}: {compressed_count} compressed nodes")
    
    # 3. Set task-related parameters
    target_node_type = task.entity_table
    
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        
        # Calculate class weights for balanced loss
        if target_node_type in compressed_labels:
            y_labels = compressed_labels[target_node_type]
            pos_count = (y_labels == 1).sum().item()
            neg_count = (y_labels == 0).sum().item()
            
            if pos_count > 0 and neg_count > 0:
                # Weight inversely proportional to class frequency
                pos_weight = torch.tensor(neg_count / pos_count, device=device)
                loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"Using weighted BCE loss: pos_weight={pos_weight:.3f} (neg:pos = {neg_count}:{pos_count})")
            else:
                loss_fn = BCEWithLogitsLoss()
        else:
            loss_fn = BCEWithLogitsLoss()
            
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
        loss_fn = BCEWithLogitsLoss()
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")
    
    # 4. Create compressed heterogeneous GNN model
    if target_node_type in compressed_data.node_types:
        feature_dim = compressed_data[target_node_type].x.size(1)
    else:
        feature_dim = args.channels
    
    # Filter edge types to only include those with actual edges and valid node types
    valid_edge_types = []
    for edge_type in compressed_data.edge_types:
        edge_index = compressed_data[edge_type].edge_index
        if edge_index is not None and edge_index.size(1) > 0:
            # Check if both source and target node types have valid features
            src_type, _, dst_type = edge_type
            src_features = compressed_data[src_type].x if src_type in compressed_data.node_types else None
            dst_features = compressed_data[dst_type].x if dst_type in compressed_data.node_types else None
            
            if (src_features is not None and src_features.size(0) > 0 and 
                dst_features is not None and dst_features.size(0) > 0):
                valid_edge_types.append(edge_type)
    
    # Filter node types to only include those with actual features
    valid_node_types = []
    for node_type in compressed_data.node_types:
        node_features = compressed_data[node_type].x
        if node_features is not None and node_features.size(0) > 0:
            valid_node_types.append(node_type)
    
    model = CompressedHeteroGNN(
        node_types=valid_node_types,
        edge_types=valid_edge_types,
        input_dim=feature_dim,
        hidden_dim=args.channels,
        output_dim=out_channels,
        aggr=args.aggr,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Created CompressedHeteroGNN model following RelBench approach:")
    print(f"  Node types: {len(valid_node_types)} valid types (out of {len(compressed_data.node_types)} total)")
    print(f"  Edge types: {len(valid_edge_types)} valid types (out of {len(compressed_data.edge_types)} total)")
    print(f"  Input dim: {feature_dim}")
    print(f"  Hidden dim: {args.channels}")
    print(f"  Output dim: {out_channels}")
    print(f"  Aggregation: {args.aggr}")
    print(f"  Layers: {args.num_layers}")
    
    # 5. Optimizer with L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Added weight decay
    
    # 6. Prepare training data
    if target_node_type not in compressed_data.node_types:
        raise ValueError(f"Target node type {target_node_type} not found in compressed data. Available: {list(compressed_data.node_types)}")
    
    if target_node_type not in compressed_labels:
        raise ValueError(f"Labels for {target_node_type} not found")
    
    # Prepare data for heterogeneous training
    x_dict = {}
    for node_type in compressed_data.node_types:
        node_features = compressed_data[node_type].x
        if node_features is not None and node_features.size(0) > 0:
            x_dict[node_type] = node_features.to(device)
        else:
            print(f"Warning: Node type {node_type} has no features, skipping")
    
    edge_index_dict = {}
    for edge_type in compressed_data.edge_types:
        edge_index = compressed_data[edge_type].edge_index
        if edge_index is not None and edge_index.size(1) > 0:
            # Check if both source and target node types exist in x_dict
            src_type, _, dst_type = edge_type
            if src_type in x_dict and dst_type in x_dict:
                edge_index_dict[edge_type] = edge_index.to(device)
    
    y = compressed_labels[target_node_type].to(device)
    
    # Simple train/validation split (80/20)
    num_nodes = y.size(0)
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = indices[:train_size]
    val_mask = indices[train_size:]
    
    print(f"Training on {len(train_mask)} nodes, validating on {len(val_mask)} nodes")
    print(f"Using {len(edge_index_dict)} valid edge types: {list(edge_index_dict.keys())}")
    
    # 7. Training loop with early stopping
    best_val_metric = float('inf') if task.task_type == TaskType.REGRESSION else 0
    best_epoch = 0
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass using heterogeneous model (much simpler!)
        pred = model(x_dict, edge_index_dict, target_node_type, train_mask)
        
        # Calculate training loss
        if len(pred.shape) > 1 and pred.shape[1] == 1:
            pred = pred.squeeze()
        train_loss = loss_fn(pred, y[train_mask])  # pred already masked
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate_model(model, compressed_data, target_node_type, y, task.task_type, val_mask)
            
            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                val_metric = val_metrics.get('auc', 0)
                is_better = val_metric > best_val_metric
                metric_name = "AUC"
            elif task.task_type == TaskType.REGRESSION:
                val_metric = val_metrics.get('mae', float('inf'))
                is_better = val_metric < best_val_metric
                metric_name = "MAE"
            else:
                val_metric = val_metrics.get('f1', 0)
                is_better = val_metric > best_val_metric
                metric_name = "F1"
            
            if is_better:
                best_val_metric = val_metric
                best_epoch = epoch
                patience_counter = 0  # Reset patience counter
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metric': val_metric,
                    'args': args,
                }, f'./saved_models/compressed_gnn_{args.dataset}_{args.task}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (patience: {patience})")
                    break
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val {metric_name}: {val_metric:.4f} | Best: {best_val_metric:.4f} (Epoch {best_epoch}) | Patience: {patience_counter}/{patience}")
    
    print(f"\nTraining completed!")
    print(f"Best validation {metric_name}: {best_val_metric:.4f} at epoch {best_epoch}")
    
    return model, compressed_data, compressed_labels


def main():
    """
    Main function
    """
    print("=== Step5: Training Compressed Graph Model ===")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Compression rate: {args.compression_rate}")
    print(f"Device: {device}")
    
    # Create save directories
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./saved_embeddings", exist_ok=True)
    
    # Train compressed model
    model, compressed_data, compressed_labels = train_compressed_model()
    
    print("Step5 completed successfully!")


if __name__ == "__main__":
    main()