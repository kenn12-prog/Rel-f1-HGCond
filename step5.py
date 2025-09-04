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
parser.add_argument("--compression_rate", type=float, default=0.05)
parser.add_argument("--use_convolution", action="store_true", help="Use GCN layers instead of MLP")
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

class CompressedGNN(nn.Module):
    """
    Simplified GNN model for compressed graphs
    Optimized specifically for small compressed graphs
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 use_convolution: bool = True, num_layers: int = 2):
        super().__init__()
        
        self.use_convolution = use_convolution
        self.num_layers = num_layers
        
        if use_convolution:
            # Use GCN for graph convolution
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            if num_layers > 1:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.classifier = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
            
        else:
            # Pure MLP since SGC already aggregated neighbor information
            layers = []
            
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, edge_index=None, batch=None):
        if self.use_convolution:
            # Convolution version - suitable for compressed graphs
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:  # No activation on last layer
                    x = F.relu(x)
                    x = self.dropout(x)
            
            # For node-level prediction, use node features directly
            return self.classifier(x)
        else:
            # MLP version - SGC already aggregated neighbor information
            return self.mlp(x)


def load_step3_results():
    """
    Load SGC convolution results and original batch data from step3
    """
    print("Loading step3 SGC convolution results...")
    
    sgc_path = Path(f"./saved_embeddings/{args.dataset}_{args.task}_sgc_features_step3.pth")
    if not sgc_path.exists():
        raise FileNotFoundError(f"SGC features not found at {sgc_path}. Please run step3 first.")
    
    step3_data = torch.load(sgc_path, map_location='cpu', weights_only=False)
    
    # step3 data contains: 'convolved_batches', 'metapaths', 'node_types', 'edge_types', 'args'
    # Use convolved_batches as original batch data
    original_batches = step3_data.get('convolved_batches', [])
    
    print(f"Loaded step3 results:")
    print(f"  Node types: {step3_data.get('node_types', [])}")
    print(f"  Number of batches: {len(original_batches)}")
    
    # Count total nodes for each node type (from all batches)
    node_counts = {}
    if original_batches:
        for node_type in step3_data.get('node_types', []):
            total_nodes = 0
            for batch in original_batches:
                if node_type in batch and 'embeddings' in batch[node_type]:
                    total_nodes += batch[node_type]['embeddings'].shape[0]
            node_counts[node_type] = total_nodes
            if total_nodes > 0:
                print(f"  {node_type}: {total_nodes} nodes total")
    
    # Add node_counts to step3_data for compatibility with subsequent code
    step3_data['node_counts'] = node_counts
    
    return step3_data, original_batches


def aggregate_batch_data_to_global(step3_data, original_batches):
    """
    Aggregate step3 batch data into global features to match step4 expected format
    Simplified version: fast concatenation for compressed graph training
    """
    print("Aggregating batch data to global features...")
    
    node_types = step3_data.get('node_types', [])
    
    global_features = {}
    global_node_mapping = {}
    node_counts = {}
    
    for node_type in node_types:
        all_features = []
        
        # Quickly collect all features
        for batch in original_batches:
            if node_type in batch and 'embeddings' in batch[node_type]:
                all_features.append(batch[node_type]['embeddings'])
        
        if all_features:
            # Directly concatenate features
            combined_features = torch.cat(all_features, dim=0)
            
            # Create simple continuous mapping (0, 1, 2, ...)
            num_nodes = combined_features.shape[0]
            node_mapping = {i: i for i in range(num_nodes)}
            
            global_features[node_type] = combined_features
            global_node_mapping[node_type] = node_mapping
            node_counts[node_type] = num_nodes
            
            print(f"  {node_type}: {num_nodes} nodes, feature dim: {combined_features.shape[1]}")
        else:
            # Empty node type
            global_features[node_type] = torch.empty(0, 128)
            global_node_mapping[node_type] = {}
            node_counts[node_type] = 0
            print(f"  {node_type}: 0 nodes (empty)")
    
    # Update step3_data format to match step4 expectations
    step3_data['convolved_features'] = global_features
    step3_data['global_node_mapping'] = global_node_mapping
    step3_data['node_counts'] = node_counts
    
    return step3_data


def evaluate_model(model, compressed_data, target_node_type, labels, task_type, split_mask=None):
    """
    Evaluate compressed graph model performance
    """
    model.eval()
    with torch.no_grad():
        if target_node_type not in compressed_data.node_types:
            return {"error": f"Target node type {target_node_type} not found"}
        
        # Get features and labels
        x = compressed_data[target_node_type].x.to(device)
        y_true = labels.to(device)
        
        if split_mask is not None:
            x = x[split_mask]
            y_true = y_true[split_mask]
        
        # Get edge information (if using convolution)
        edge_index = None
        if hasattr(model, 'use_convolution') and model.use_convolution:
            # Find edges containing target_node_type
            found_edge = False
            for edge_type in compressed_data.edge_types:
                src_type, rel_type, dst_type = edge_type
                if src_type == target_node_type and dst_type == target_node_type:
                    edge_index = compressed_data[edge_type].edge_index.to(device)
                    found_edge = True
                    break
            
            # If no self-loop edges found, create simple self-loop edges
            if not found_edge:
                num_nodes = x.size(0)
                # Create self-loop edges: each node connects to itself
                self_loop_edges = torch.arange(num_nodes, device=device)
                edge_index = torch.stack([self_loop_edges, self_loop_edges], dim=0)
        
        # Forward pass
        pred = model(x, edge_index)
        
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
    
    # 2. Load step3 results
    step3_data, original_batches = load_step3_results()
    
    # 3. Aggregate batch data to global features (match step4 expected format)
    step3_data = aggregate_batch_data_to_global(step3_data, original_batches)
    
    # 4. Create graph compressor
    compressor = GraphCompressor(compression_rate=args.compression_rate)
    
    # 5. Use GraphCompressor's create_compressed_dataset static method
    compressed_data, compressed_labels = GraphCompressor.create_compressed_dataset(
        step3_data, original_batches, task, compressor
    )
    
    print(f"Graph compression completed:")
    for node_type in compressed_data.node_types:
        if node_type in step3_data['node_counts']:
            original_count = step3_data['node_counts'][node_type]
            compressed_count = compressed_data[node_type].num_nodes
            print(f"  {node_type}: {original_count} -> {compressed_count} ({compressed_count/original_count:.3f})")
    
    # 6. Set task-related parameters
    target_node_type = task.entity_table
    
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
        loss_fn = BCEWithLogitsLoss()
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")
    
    # 7. Create compressed GNN model
    if target_node_type in compressed_data.node_types:
        feature_dim = compressed_data[target_node_type].x.size(1)
    else:
        feature_dim = args.channels
    
    model = CompressedGNN(
        input_dim=feature_dim,
        hidden_dim=args.channels,
        output_dim=out_channels,
        use_convolution=args.use_convolution,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Created CompressedGNN model:")
    print(f"  Input dim: {feature_dim}")
    print(f"  Hidden dim: {args.channels}")
    print(f"  Output dim: {out_channels}")
    print(f"  Use convolution: {args.use_convolution}")
    print(f"  Layers: {args.num_layers}")
    
    # 8. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 9. Prepare training data
    if target_node_type not in compressed_data.node_types:
        raise ValueError(f"Target node type {target_node_type} not found in compressed data. Available: {list(compressed_data.node_types)}")
    
    if target_node_type not in compressed_labels:
        raise ValueError(f"Labels for {target_node_type} not found")
    
    # Get features and labels
    x = compressed_data[target_node_type].x.to(device)
    y = compressed_labels[target_node_type].to(device)
    
    # Simple train/validation split (80/20)
    num_nodes = x.size(0)
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = indices[:train_size]
    val_mask = indices[train_size:]
    
    print(f"Training on {len(train_mask)} nodes, validating on {len(val_mask)} nodes")
    
    # 10. Training loop
    best_val_metric = float('inf') if task.task_type == TaskType.REGRESSION else 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Get edge information (if using convolution)
        edge_index = None
        if args.use_convolution:
            # Find any edges containing target_node_type, create self-loops if none found
            found_edge = False
            for edge_type in compressed_data.edge_types:
                src_type, rel_type, dst_type = edge_type
                if src_type == target_node_type and dst_type == target_node_type:
                    edge_index = compressed_data[edge_type].edge_index.to(device)
                    found_edge = True
                    break
            
            # If no self-loop edges found, create simple self-loop edges
            if not found_edge:
                num_nodes = x.size(0)
                # Create self-loop edges: each node connects to itself
                self_loop_edges = torch.arange(num_nodes, device=device)
                edge_index = torch.stack([self_loop_edges, self_loop_edges], dim=0)
        
        # Forward pass
        pred = model(x, edge_index)
        
        # Calculate training loss
        if len(pred.shape) > 1 and pred.shape[1] == 1:
            pred = pred.squeeze()
        train_loss = loss_fn(pred[train_mask], y[train_mask])
        
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
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metric': val_metric,
                    'args': args,
                }, f'./saved_models/compressed_gnn_{args.dataset}_{args.task}_best.pth')
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val {metric_name}: {val_metric:.4f} | Best: {best_val_metric:.4f} (Epoch {best_epoch})")
    
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
    print(f"Use convolution: {args.use_convolution}")
    print(f"Device: {device}")
    
    # Create save directories
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./saved_embeddings", exist_ok=True)
    
    # Train compressed model
    model, compressed_data, compressed_labels = train_compressed_model()
    
    print("Step5 completed successfully!")


if __name__ == "__main__":
    main()