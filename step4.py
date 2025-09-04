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
parser.add_argument("--compression_rate", type=float, default=0.005)
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
        """
        compressed_features = {}
        compressed_labels = {}
        cluster_assignments = {}
        
        for node_type, features in convolved_features.items():
            # Skip empty node types
            if features.size(0) == 0:
                print(f"  Skipping {node_type}: 0 nodes")
                continue
                
            # Perform clustering (regardless of label availability)
            compressed_count = max(1, int(features.size(0) * self.compression_rate))
            cluster_centers, cluster_labels = self.cluster_nodes(features, compressed_count)
            
            compressed_features[node_type] = cluster_centers
            cluster_assignments[node_type] = cluster_labels
            
            # Compress labels if available
            if node_type in labels_dict:
                original_labels = labels_dict[node_type]
                
                # Assign labels to each cluster using majority voting
                cluster_node_labels = torch.zeros(compressed_count)

                for cluster_id in range(compressed_count):
                    # Find all original nodes belonging to this cluster
                    mask = (cluster_labels == cluster_id)
                    if mask.sum() > 0:
                        cluster_original_labels = original_labels[mask]
                        # Majority voting
                        if len(cluster_original_labels) > 0:
                            cluster_node_labels[cluster_id] = cluster_original_labels.mode()[0]
                
                compressed_labels[node_type] = cluster_node_labels
            else:
                # If no labels available, create zero labels (but keep compressed features)
                compressed_labels[node_type] = torch.zeros(compressed_count)
        
        return compressed_features, compressed_labels, cluster_assignments
    
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

        
