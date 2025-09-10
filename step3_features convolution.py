# Step3: SGC convolution for heterogeneous graph propagation

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import numpy as np
from torch_sparse import SparseTensor

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-dnf")
parser.add_argument("--num_layers", type=int, default=2, help="SGC convolution layers")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(args.seed)

class HeteroSGCConv:
    """
    Heterogeneous SGC convolution implementation
    Supports multi-metapath propagation without trainable parameters
    Similar to FreeHGC graph propagation mechanism
    """
    def __init__(self, num_layers: int = 2):
        self.num_layers = num_layers
    
    def generate_metapaths(self, node_types: List[str], edge_types: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
        """
        Generate length-3 metapaths for each node type
        Ensures start and end node types are consistent (e.g., A->B->A)
        Similar to FreeHGC metapath generation logic
        """
        metapaths = defaultdict(list)
        
        # Build adjacency dictionary
        adjacency = defaultdict(list)
        for src, rel, dst in edge_types:
            adjacency[src].append((rel, dst))
        
        # Generate metapaths for each node type
        for node_type in node_types:
            # Length-3 paths: A -> B -> A (e.g., PAP, APA)
            if node_type in adjacency:
                for rel1, mid_type in adjacency[node_type]:
                    if mid_type in adjacency:
                        for rel2, end_type in adjacency[mid_type]:
                            if end_type == node_type:  # Ensure start and end consistency
                                metapath = f"{node_type}{rel1}{mid_type}{rel2}{end_type}"
                                metapaths[node_type].append(metapath)
        
        return metapaths
    
    def _build_node_mapping(self, batch: Dict, node_types: List[str]) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        Build node mapping for a single batch
        Maps global node IDs to local batch indices
        """
        node_mapping = {}
        node_counts = {}
        
        for node_type in node_types:
            if node_type in batch and 'node_indices' in batch[node_type]:
                indices = batch[node_type]['node_indices']
                unique_indices = torch.unique(indices)
                node_mapping[node_type] = {idx.item(): i for i, idx in enumerate(unique_indices)}
                node_counts[node_type] = len(unique_indices)
            else:
                node_mapping[node_type] = {}
                node_counts[node_type] = 0
        
        return node_mapping, node_counts
    
    def _build_base_adjacency_matrices(self, batch: Dict, edge_types: List[Tuple[str, str, str]], 
                                     node_mapping: Dict[str, Dict], node_counts: Dict[str, int]) -> Dict[str, SparseTensor]:
        """
        Build base adjacency matrices (e.g., AP, PA)
        Creates sparse matrices for each edge type
        """
        base_adj = {}
        
        edge_index_dict = batch['edge_index_dict']
        
        for src_type, rel_type, dst_type in edge_types:
            edge_key = f"{src_type}{dst_type}"
            
            # Find matching edge type
            matching_edge_type = None
            for edge_type_tuple, edge_index in edge_index_dict.items():
                if (isinstance(edge_type_tuple, tuple) and len(edge_type_tuple) == 3 and
                    edge_type_tuple[0] == src_type and edge_type_tuple[2] == dst_type):
                    matching_edge_type = edge_type_tuple
                    break
            
            if matching_edge_type is not None:
                edge_index = edge_index_dict[matching_edge_type]
                
                if edge_index.size(1) > 0:  # Ensure edges exist
                    # Map edge indices to local indices
                    src_indices = edge_index[0]
                    dst_indices = edge_index[1]
                    
                    # Create mapped edge indices
                    mapped_src = []
                    mapped_dst = []
                    
                    for i in range(len(src_indices)):
                        src_idx = src_indices[i].item()
                        dst_idx = dst_indices[i].item()
                        
                        if (src_idx in node_mapping[src_type] and 
                            dst_idx in node_mapping[dst_type]):
                            mapped_src.append(node_mapping[src_type][src_idx])
                            mapped_dst.append(node_mapping[dst_type][dst_idx])
                    
                    if mapped_src and mapped_dst:
                        row = torch.tensor(mapped_src, dtype=torch.long)
                        col = torch.tensor(mapped_dst, dtype=torch.long)
                        
                        # Create sparse matrix
                        adj = SparseTensor(
                            row=row,
                            col=col,
                            sparse_sizes=(node_counts[src_type], node_counts[dst_type])
                        )
                        
                        base_adj[edge_key] = adj
                        
                        # Create reverse adjacency matrix for different node types
                        if src_type != dst_type:
                            reverse_key = f"{dst_type}{src_type}"
                            reverse_adj = SparseTensor(
                                row=col,
                                col=row,
                                sparse_sizes=(node_counts[dst_type], node_counts[src_type])
                            )
                            base_adj[reverse_key] = reverse_adj
        
        return base_adj
    
    def _parse_metapath(self, metapath: str, known_node_types: List[str]) -> List[str]:
        """
        Parse metapath string to extract node type sequence
        Example: "resultsrelconstructor_standingsrelresults" -> ['results', 'constructor_standings', 'results']
        """
        # Use 'rel' as separator to split metapath
        parts = metapath.split('rel')
        
        # Filter out empty strings, keep actual node types
        node_types = [part for part in parts if part and part != '']
        
        return node_types
    
    def build_metapath_adjacency_matrices_single_batch(self, batch: Dict, node_types: List[str], 
                                                     edge_types: List[Tuple[str, str, str]], 
                                                     metapaths: Dict[str, List[str]]) -> Tuple[Dict[str, SparseTensor], Dict[str, Dict], Dict[str, int]]:
        """
        Build metapath adjacency matrices for a single batch
        Combines base adjacency matrices via matrix multiplication
        """
        # 1. Build node mapping
        node_mapping, node_counts = self._build_node_mapping(batch, node_types)
        
        # 2. Build base adjacency matrices
        base_adj = self._build_base_adjacency_matrices(batch, edge_types, node_mapping, node_counts)
        
        # 3. Compose metapath adjacency matrices
        metapath_adj = {}
        
        for node_type, paths in metapaths.items():
            for metapath in paths:
                # Parse metapath
                path_nodes = self._parse_metapath(metapath, node_types)
                
                if len(path_nodes) >= 3:
                    # Build complete metapath matrix
                    result_adj = None
                    
                    for i in range(len(path_nodes) - 1):
                        edge_key = f"{path_nodes[i]}{path_nodes[i+1]}"
                        
                        if edge_key in base_adj:
                            if result_adj is None:
                                result_adj = base_adj[edge_key]
                            else:
                                # Matrix multiplication to compose metapath
                                result_adj = result_adj @ base_adj[edge_key]
                    
                    if result_adj is not None:
                        metapath_adj[metapath] = result_adj
        
        return metapath_adj, node_mapping, node_counts

    def sgc_propagate(self, features: Dict[str, torch.Tensor], 
                     adj_matrices: Dict[str, SparseTensor],
                     metapaths: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        SGC-style graph convolution propagation
        """
        print(f"Starting SGC propagation with {self.num_layers} layers...")
        
        # Normalize adjacency matrices
        normalized_adj = {}
        for key, adj in adj_matrices.items():
            # Row normalization
            row_sum = torch.sparse.sum(adj.to_torch_sparse_coo_tensor(), dim=1).to_dense()
            row_sum = torch.clamp(row_sum, min=1e-12)  # Avoid division by zero
            
            # Create inverse degree matrix
            deg_inv = 1.0 / row_sum
            deg_inv = deg_inv.unsqueeze(1)
            
            # Normalize values
            normalized_values = adj.storage.value() if adj.storage.value() is not None else torch.ones(adj.nnz())
            normalized_values = normalized_values / row_sum[adj.storage.row()]
            
            normalized_adj[key] = SparseTensor(
                row=adj.storage.row(),
                col=adj.storage.col(),
                value=normalized_values,
                sparse_sizes=adj.sparse_sizes()
            )
        
        # Multi-layer propagation
        current_features = {k: v.clone() for k, v in features.items()}
    
        for layer in range(self.num_layers):
            next_features = {}
            
            for node_type in features.keys():
                if node_type not in metapaths:
                    next_features[node_type] = current_features[node_type]
                    continue
                
                aggregated_features = []
                
                # Use pre-composed metapath adjacency matrices
                for metapath in metapaths[node_type]:
                    if metapath in normalized_adj:
                        # Complete metapath propagation in one step
                        propagated = normalized_adj[metapath] @ current_features[node_type]
                        aggregated_features.append(propagated)
                
                # Average aggregated features or keep original features
                if aggregated_features:
                    next_features[node_type] = torch.stack(aggregated_features).mean(dim=0)
                else:
                    next_features[node_type] = current_features[node_type]
            
            current_features = next_features
        
        return current_features

def extract_graph_schema(all_batches: List[Dict]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Extract graph schema information (node types and edge types) from all batches
    """
    node_types = set()
    edge_types = set()
    
    for batch in all_batches:
        for key, value in batch.items():
            if isinstance(value, dict):
                # Check if it's node data
                if 'embeddings' in value or 'node_indices' in value:
                    # Extract node type from key (assuming key is node type)
                    node_types.add(key)
                
                # Check if it's edge data
                if 'edge_index' in value:
                    # Try to parse edge type from key
                    # Assume edge key format like "author_paper" or "paper_author"
                    parts = key.split('_')
                    if len(parts) >= 2:
                        src_type = parts[0]
                        dst_type = parts[-1]
                        rel_type = '_'.join(parts[1:-1]) if len(parts) > 2 else 'rel'
                        edge_types.add((src_type, rel_type, dst_type))
                    else:
                        # If parsing fails, use default format
                        edge_types.add((key, 'rel', key))
    
    # If no explicit edge types found, generate default edge types from node types
    if not edge_types and len(node_types) >= 2:
        node_list = list(node_types)
        for i, src in enumerate(node_list):
            for j, dst in enumerate(node_list):
                if i != j:  # Edges between different node types
                    edge_types.add((src, 'rel', dst))
    
    return list(node_types), list(edge_types)

def load_embeddings_and_process():
    """
    Load embeddings from step2 and apply SGC convolution processing
    """
    print("Loading embeddings from step2...")
    
    embedding_path = Path(f"./saved_embeddings/{args.dataset}_{args.task}_embeddings_step2.pth")
    all_batches = torch.load(embedding_path, map_location='cpu', weights_only=False)
    print(f"Loaded {len(all_batches)} batches of embeddings")
    
    # Extract global graph schema information
    node_types, edge_types = extract_graph_schema(all_batches)
    
    # Initialize SGC convolution
    sgc_conv = HeteroSGCConv(num_layers=args.num_layers)
    
    # Generate metapaths
    metapaths = sgc_conv.generate_metapaths(node_types, edge_types)
    print("Generated metapaths:")
    for node_type, paths in metapaths.items():
        print(f"  {node_type}: {paths}")
    
    # Collect convolution results from all batches
    all_convolved_batches = []
    
    # Process each batch sequentially
    for batch_idx, batch in enumerate(all_batches):
        print(f"Processing batch {batch_idx + 1}/{len(all_batches)}")
        
        # 1. Build adjacency matrices for current batch
        batch_metapath_adj, batch_node_mapping, batch_node_counts = sgc_conv.build_metapath_adjacency_matrices_single_batch(
            batch, node_types, edge_types, metapaths
        )
        
        # 2. Organize current batch features
        batch_features = {}
        for node_type in node_types:
            if node_type in batch and 'embeddings' in batch[node_type]:
                batch_features[node_type] = batch[node_type]['embeddings']
        
        # 3. Apply SGC convolution to current batch
        if batch_features and batch_metapath_adj:
            convolved_batch_features = sgc_conv.sgc_propagate(
                batch_features, batch_metapath_adj, metapaths
            )
            
            # 4. Save convolved batch data (preserve original node_indices)
            convolved_batch = {}
            for node_type in node_types:
                if (node_type in convolved_batch_features and 
                    node_type in batch and 'node_indices' in batch[node_type]):
                    convolved_batch[node_type] = {
                        'embeddings': convolved_batch_features[node_type],
                        'node_indices': batch[node_type]['node_indices']
                    }
            
            # Preserve edge_index_dict for subsequent steps
            if 'edge_index_dict' in batch:
                convolved_batch['edge_index_dict'] = batch['edge_index_dict']
            
            all_convolved_batches.append(convolved_batch)
    
    # 5. Save results
    save_data = {
        'convolved_batches': all_convolved_batches,
        'metapaths': metapaths,
        'node_types': node_types,
        'edge_types': edge_types,
        'args': vars(args)
    }
    
    output_path = Path("./saved_embeddings") / f"{args.dataset}_{args.task}_sgc_features_step3.pth"
    torch.save(save_data, output_path)
    
    print(f"Processed {len(all_convolved_batches)} batches with SGC convolution")
    return all_convolved_batches

if __name__ == "__main__":
    print(f"Starting SGC convolution with {args.num_layers} layers...")
    load_embeddings_and_process()
    print("SGC convolution completed!")
