# Step4: Graph compression using KMeans clustering

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
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-dnf")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--compression_rate", type=float, default=0.1)
parser.add_argument("--force", action="store_true", help="Force recreate compressed graph even if it exists")
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

class GraphCompressor:
    """
    Graph compression method based on FreeHGC approach
    Uses KMeans clustering for node compression
    """
    def __init__(self, compression_rate: float = 0.05):
        self.compression_rate = compression_rate
    
    def cluster_nodes(self, features: torch.Tensor, n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster nodes using KMeans, similar to FreeHGC's KCenter method
        """
        # Handle edge cases
        if features.size(0) == 0:
            # Return empty tensors
            return torch.empty(0, features.size(1)), torch.empty(0, dtype=torch.long)
        
        if n_clusters <= 0:
            n_clusters = 1
        
        # Ensure cluster count doesn't exceed node count and is at least 1
        n_clusters = max(1, min(n_clusters, features.size(0)))
        
        # Use MiniBatchKMeans for large node types, KMeans for small ones
        if features.size(0) > 50000:  # Use fast algorithm for >50k nodes
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=1024,  # Batch size for processing
                max_iter=100
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                n_init=1,
                max_iter=100,
                algorithm='lloyd'
            )
        features_np = features.detach().cpu().numpy()
        cluster_labels = kmeans.fit_predict(features_np)
        
        # Get cluster centers
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(features.device)
        cluster_labels = torch.from_numpy(cluster_labels).to(features.device)
        
        return cluster_centers, cluster_labels
    
    def compress_graph_with_labels(self, convolved_features, labels_dict, global_node_mapping):
        """
        Compress graph and assign labels to clustered nodes
        For target node types with labels, perform clustering within each class
        and distribute clusters proportionally based on class sizes
        """
        compressed_features = {}
        compressed_labels = {}
        cluster_assignments = {}
        
        for node_type, features in convolved_features.items():
            # Skip empty node types
            if features.size(0) == 0:
                print(f"  Skipping {node_type}: 0 nodes")
                continue
                
            total_compressed_count = max(1, int(features.size(0) * self.compression_rate))
            
            # Check if this node type has labels (target node type)
            if node_type in labels_dict:
                original_labels = labels_dict[node_type]
                
                # Perform class-wise clustering for target nodes
                compressed_centers, compressed_node_labels, cluster_labels = self._cluster_by_class(
                    features, original_labels, total_compressed_count
                )
                
                compressed_features[node_type] = compressed_centers
                compressed_labels[node_type] = compressed_node_labels
                cluster_assignments[node_type] = cluster_labels
                
                print(f"  {node_type}: {features.size(0)} -> {compressed_centers.size(0)} nodes (class-wise clustering)")
                
            else:
                # For non-target node types, use regular clustering
                cluster_centers, cluster_labels = self.cluster_nodes(features, total_compressed_count)
                
                compressed_features[node_type] = cluster_centers
                cluster_assignments[node_type] = cluster_labels
                compressed_labels[node_type] = torch.zeros(total_compressed_count)
                
                print(f"  {node_type}: {features.size(0)} -> {cluster_centers.size(0)} nodes (regular clustering)")
        
        return compressed_features, compressed_labels, cluster_assignments
    
    def _cluster_by_class(self, features: torch.Tensor, labels: torch.Tensor, total_clusters: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform clustering within each class and distribute clusters proportionally
        
        Args:
            features: Node features [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            total_clusters: Total number of clusters to create
            
        Returns:
            compressed_centers: Cluster centers [total_clusters, feature_dim]
            compressed_labels: Labels for each cluster [total_clusters]
            cluster_assignments: Original node to cluster mapping [num_nodes]
        """
        # Get unique classes and their counts
        unique_classes = torch.unique(labels)
        class_counts = {}
        class_indices = {}
        
        for class_label in unique_classes:
            mask = (labels == class_label)
            class_counts[class_label.item()] = mask.sum().item()
            class_indices[class_label.item()] = torch.where(mask)[0]
        
        total_nodes = features.size(0)
        
        # Allocate clusters proportionally to each class
        class_cluster_counts = {}
        allocated_clusters = 0
        
        # Sort classes by size for better allocation
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (class_label, count) in enumerate(sorted_classes):
            if i == len(sorted_classes) - 1:  # Last class gets remaining clusters
                class_cluster_counts[class_label] = max(1, total_clusters - allocated_clusters)
            else:
                # Proportional allocation, but at least 1 cluster per class
                proportion = count / total_nodes
                allocated = max(1, int(total_clusters * proportion))
                class_cluster_counts[class_label] = allocated
                allocated_clusters += allocated
        
        print(f"    Class distribution: {dict(class_counts)}")
        print(f"    Cluster allocation: {class_cluster_counts}")
        
        # Perform clustering within each class
        all_centers = []
        all_cluster_labels = []
        cluster_assignments = torch.zeros(features.size(0), dtype=torch.long)
        current_cluster_id = 0
        
        for class_label in unique_classes:
            class_label_item = class_label.item()
            class_mask = class_indices[class_label_item]
            class_features = features[class_mask]
            n_clusters = class_cluster_counts[class_label_item]
            
            if class_features.size(0) == 0:
                continue
                
            # Cluster within this class
            if n_clusters >= class_features.size(0):
                # If we need more clusters than nodes, use each node as a cluster
                class_centers = class_features
                class_cluster_labels = torch.arange(class_features.size(0))
            else:
                class_centers, class_cluster_labels = self.cluster_nodes(class_features, n_clusters)
            
            # Store results
            all_centers.append(class_centers)
            all_cluster_labels.extend([class_label_item] * class_centers.size(0))
            
            # Update cluster assignments for original nodes
            for i, original_idx in enumerate(class_mask):
                cluster_assignments[original_idx] = current_cluster_id + class_cluster_labels[i]
            
            current_cluster_id += class_centers.size(0)
        
        # Combine all cluster centers and labels
        if all_centers:
            compressed_centers = torch.cat(all_centers, dim=0)
            compressed_node_labels = torch.tensor(all_cluster_labels, dtype=labels.dtype, device=labels.device)
        else:
            # Fallback case
            compressed_centers = torch.zeros(1, features.size(1), device=features.device)
            compressed_node_labels = torch.zeros(1, dtype=labels.dtype, device=labels.device)
            cluster_assignments = torch.zeros(features.size(0), dtype=torch.long)
        
        return compressed_centers, compressed_node_labels, cluster_assignments
    
    def build_compressed_edges(self, cluster_assignments: Dict[str, torch.Tensor],
                              original_edge_data: List[Dict]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build compressed graph edges
        If any nodes in clusters are connected, then clusters are connected
        """
        print("Building compressed graph edges...")
        
        compressed_edges = defaultdict(set)
        
        # Process edges from original batch data
        for batch in original_edge_data:
            if 'edge_index_dict' not in batch:
                continue
                
            for edge_type, edge_index in batch['edge_index_dict'].items():
                if isinstance(edge_type, tuple) and len(edge_type) == 3:
                    src_type, rel_type, dst_type = edge_type
                    
                    if src_type not in cluster_assignments or dst_type not in cluster_assignments:
                        continue
                    
                    # Map original edges to cluster edges
                    for i in range(edge_index.size(1)):
                        src_node = edge_index[0, i].item()
                        dst_node = edge_index[1, i].item()
                        
                        # Get cluster labels
                        if (src_node < cluster_assignments[src_type].size(0) and 
                            dst_node < cluster_assignments[dst_type].size(0)):
                            
                            src_cluster = cluster_assignments[src_type][src_node].item()
                            dst_cluster = cluster_assignments[dst_type][dst_node].item()
                            
                            compressed_edges[edge_type].add((src_cluster, dst_cluster))
        
        # Convert to tensor format
        compressed_edge_dict = {}
        for edge_type, edge_set in compressed_edges.items():
            if edge_set:
                edges = list(edge_set)
                src_indices, dst_indices = zip(*edges)
                compressed_edge_dict[edge_type] = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        return compressed_edge_dict

    @staticmethod
    def create_compressed_dataset(step3_data: Dict,
                                original_batches: List[Dict], 
                                task,
                                compressor: 'GraphCompressor') -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
        """
        Create compressed graph dataset using defined GraphCompressor methods
        """
        print("Creating compressed dataset using GraphCompressor...")
        
        # 1. Prepare label data (from task)
        labels_dict = {}
        target_node_type = task.entity_table
        
        try:
            # Get labels from training data
            train_table = task.get_table("train")
            label_mapping = {}
            for idx, row in train_table.df.iterrows():
                node_id = row[task.entity_col]
                label = row[task.target_col]
                label_mapping[node_id] = label
            
            # Create label tensor for target node type
            if target_node_type in step3_data['convolved_features']:
                # Map labels to correct nodes using global_node_mapping
                global_mapping = step3_data['global_node_mapping'][target_node_type]
                num_nodes = step3_data['node_counts'][target_node_type]
                node_labels = torch.zeros(num_nodes)
                
                for orig_node_id, mapped_idx in global_mapping.items():
                    if orig_node_id in label_mapping:
                        node_labels[mapped_idx] = label_mapping[orig_node_id]
                
                labels_dict[target_node_type] = node_labels
                print(f"  Created labels for {num_nodes} nodes of type {target_node_type}")
            
        except Exception as e:
            print(f"Warning: Could not create labels: {e}")
            print(f"This will cause training issues. Please check the task and data.")
            # Don't create dummy labels, let subsequent code handle this issue
            pass
        
        # 2. Use GraphCompressor's compress_graph_with_labels method
        compressed_features, compressed_labels, cluster_assignments = compressor.compress_graph_with_labels(
            step3_data['convolved_features'],
            labels_dict,
            step3_data['global_node_mapping']
        )
        
        # 3. Use GraphCompressor's build_compressed_edges method
        compressed_edges = compressor.build_compressed_edges(cluster_assignments, original_batches)
        
        # 4. Create HeteroData object
        compressed_data = HeteroData()
        
        # Add compressed node features
        for node_type, features in compressed_features.items():
            compressed_data[node_type].x = features
            compressed_data[node_type].num_nodes = features.size(0)
            print(f"  {node_type}: {features.size(0)} compressed nodes, feature dim: {features.size(1)}")
            
            # Add labels if available
            if node_type in compressed_labels:
                compressed_data[node_type].y = compressed_labels[node_type]
        
        # Add compressed edge information
        edge_count = 0
        for edge_type, edge_index in compressed_edges.items():
            if len(edge_type) == 3:
                src_type, rel_type, dst_type = edge_type
                compressed_data[src_type, rel_type, dst_type].edge_index = edge_index
                edge_count += edge_index.size(1)
        
        print(f"  Total compressed edges: {edge_count}")
        
        return compressed_data, compressed_labels


def save_compressed_graph(compressed_data, compressed_labels, dataset_name, task_name, compression_rate):
    """
    Save compressed graph data to disk
    """
    save_dir = Path("./saved_compressed_graphs")
    save_dir.mkdir(exist_ok=True)
    
    filename = f"{dataset_name}_{task_name}_compressed_{compression_rate:.3f}.pth"
    save_path = save_dir / filename
    
    save_data = {
        'compressed_data': compressed_data,
        'compressed_labels': compressed_labels,
        'dataset_name': dataset_name,
        'task_name': task_name,
        'compression_rate': compression_rate,
        'node_types': list(compressed_data.node_types),
        'edge_types': list(compressed_data.edge_types),
        'creation_time': torch.tensor(time.time())
    }
    
    torch.save(save_data, save_path)
    print(f"Compressed graph saved to: {save_path}")
    return save_path


def load_compressed_graph(dataset_name, task_name, compression_rate):
    """
    Load compressed graph data from disk
    """
    save_dir = Path("./saved_compressed_graphs")
    filename = f"{dataset_name}_{task_name}_compressed_{compression_rate:.3f}.pth"
    save_path = save_dir / filename
    
    if not save_path.exists():
        raise FileNotFoundError(f"Compressed graph not found at {save_path}")
    
    print(f"Loading compressed graph from: {save_path}")
    save_data = torch.load(save_path, map_location='cpu', weights_only=False)
    
    compressed_data = save_data['compressed_data']
    compressed_labels = save_data['compressed_labels']
    
    print(f"Loaded compressed graph:")
    print(f"  Dataset: {save_data['dataset_name']}")
    print(f"  Task: {save_data['task_name']}")
    print(f"  Compression rate: {save_data['compression_rate']}")
    print(f"  Node types: {save_data['node_types']}")
    print(f"  Edge types: {len(save_data['edge_types'])} types")
    
    return compressed_data, compressed_labels


def main():
    """
    Main function for step4: Create and save compressed graph
    """
    print("=== Step4: Graph Compression ===")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Compression rate: {args.compression_rate}")
    print(f"Device: {device}")
    
    # Create save directories
    os.makedirs("./saved_compressed_graphs", exist_ok=True)
    
    # Check if compressed graph already exists
    if not args.force:
        try:
            compressed_data, compressed_labels = load_compressed_graph(
                args.dataset, args.task, args.compression_rate
            )
            print("Compressed graph already exists! Use --force to recreate.")
            return compressed_data, compressed_labels
        except FileNotFoundError:
            pass
    
    # Load step3 results
    print("\nLoading step3 results...")
    step3_data_path = Path(f"./saved_embeddings/{args.dataset}_{args.task}_sgc_features_step3.pth")
    if not step3_data_path.exists():
        raise FileNotFoundError(f"Step3 results not found at {step3_data_path}. Please run step3 first.")
    
    step3_data = torch.load(step3_data_path, map_location='cpu', weights_only=False)
    original_batches = step3_data.get('convolved_batches', [])
    
    print(f"Loaded step3 results:")
    print(f"  Node types: {step3_data.get('node_types', [])}")
    print(f"  Number of batches: {len(original_batches)}")
    
    # Aggregate batch data to global features
    print("\nAggregating batch data to global features...")
    node_types = step3_data.get('node_types', [])
    
    global_features = {}
    global_node_mapping = {}
    node_counts = {}
    
    for node_type in node_types:
        all_features = []
        
        for batch in original_batches:
            if node_type in batch and 'embeddings' in batch[node_type]:
                all_features.append(batch[node_type]['embeddings'])
        
        if all_features:
            combined_features = torch.cat(all_features, dim=0)
            num_nodes = combined_features.shape[0]
            node_mapping = {i: i for i in range(num_nodes)}
            
            global_features[node_type] = combined_features
            global_node_mapping[node_type] = node_mapping
            node_counts[node_type] = num_nodes
            
            print(f"  {node_type}: {num_nodes} nodes, feature dim: {combined_features.shape[1]}")
        else:
            global_features[node_type] = torch.empty(0, 128)
            global_node_mapping[node_type] = {}
            node_counts[node_type] = 0
    
    # Update step3_data format
    step3_data['convolved_features'] = global_features
    step3_data['global_node_mapping'] = global_node_mapping
    step3_data['node_counts'] = node_counts
    
    # Load task information
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    
    # Create graph compressor and compress graph
    print(f"\nCompressing graph with rate {args.compression_rate}...")
    compressor = GraphCompressor(compression_rate=args.compression_rate)
    
    compressed_data, compressed_labels = GraphCompressor.create_compressed_dataset(
        step3_data, original_batches, task, compressor
    )
    
    print(f"\nGraph compression completed:")
    for node_type in compressed_data.node_types:
        if node_type in step3_data['node_counts']:
            original_count = step3_data['node_counts'][node_type]
            compressed_count = compressed_data[node_type].num_nodes
            print(f"  {node_type}: {original_count} -> {compressed_count} ({compressed_count/original_count:.3f})")
    
    # Save compressed graph
    save_path = save_compressed_graph(
        compressed_data, compressed_labels, args.dataset, args.task, args.compression_rate
    )
    
    print(f"\nStep4 completed successfully!")
    print(f"Compressed graph saved to: {save_path}")
    
    return compressed_data, compressed_labels


if __name__ == "__main__":
    import time
    main()

        
