"""
Graph visualization for parameter identifiability relationships.
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def build_identifiability_graph(used_symbols, nullspace_vectors, symbol_types=None, index_range=None):
    """
    Construct a graph where nodes are variables (states/parameters/inputs).
    Edges connect variables that are part of the same unidentifiable combination.
    Isolated nodes are only identifiable if they don't appear in ANY nullspace vector.
    Parameters
    ----------
    used_symbols : list of sympy.Symbol
        Variables in the model.
    nullspace_vectors : list 
        Each vector encodes one unidentifiable direction.
    symbol_types : list of str, optional
        Type of each symbol ('state', 'param', 'input') for coloring
    index_range : tuple (start, end) or None
        Only look at indices [start:end] in the nullspace vectors
        If None, use the full vector
    """
    G = nx.Graph()
    
    # Add all nodes
    for i, s in enumerate(used_symbols):
        node_type = symbol_types[i] if symbol_types else 'unknown'
        G.add_node(str(s), node_type=node_type)

    variables_in_nullspace = set()
    
    for vec in nullspace_vectors:
        if index_range is None:
            # Use full vector - symbol index i maps to vector index i
            involved = [i for i in range(min(len(used_symbols), len(vec))) if vec[i] != 0] 
            # range(min(len(used_symbols), len(vec))) 
        else:
            # Use specified range - vector index (start_idx + i) maps to symbol index i
            start_idx, end_idx = index_range
            involved = []
            for i in range(len(used_symbols)):
                vec_idx = start_idx + i
                if vec_idx < end_idx and vec_idx < len(vec) and vec[vec_idx] != 0:
                    involved.append(i)  # i is the symbol index
        
        # Mark all involved variables as unidentifiable
        for symbol_idx in involved:
            if symbol_idx < len(used_symbols):
                variables_in_nullspace.add(str(used_symbols[symbol_idx]))
        
        # Add edges between all pairs in this nullspace vector
        for i in range(len(involved)):
            for j in range(i + 1, len(involved)):
                s1 = str(used_symbols[involved[i]])
                s2 = str(used_symbols[involved[j]])
                G.add_edge(s1, s2)

    # Mark nodes as identifiable or unidentifiable
    for node in G.nodes():
        if node in variables_in_nullspace:
            G.nodes[node]['identifiable'] = False
        else:
            G.nodes[node]['identifiable'] = True

    return G

def visualize_identifiability_graph(G, title="Variable Identifiability Graph", save_path=None):
    """
    Visualize the identifiability graph with nodes colored by variable type and identifiability.
    """
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Separate nodes by identifiability
    identifiable_nodes = [n for n in G.nodes() if G.nodes[n].get('identifiable', False)]
    unidentifiable_nodes = [n for n in G.nodes() if not G.nodes[n].get('identifiable', True)]
    
    # Color by variable type
    def get_node_color(node, base_color):
        node_type = G.nodes[node].get('node_type', 'unknown')
        if node_type == 'state':
            return 'lightblue' if base_color == 'light' else 'darkblue'
        elif node_type == 'param':
            return 'lightcoral' if base_color == 'light' else 'darkred'
        elif node_type == 'input':
            return 'lightgreen' if base_color == 'light' else 'darkgreen'
        else:
            return 'lightgray' if base_color == 'light' else 'darkgray'

    # # Draw identifiable nodes (if any)
    # if identifiable_nodes:
    #     identifiable_colors = [get_node_color(n, 'light') for n in identifiable_nodes]
    #     nx.draw_networkx_nodes(G, pos, nodelist=identifiable_nodes, 
    #                           node_color=identifiable_colors, node_size=1000, 
    #                           edgecolors='green', linewidths=3, alpha=0.8)

    # # Draw unidentifiable nodes
    # if unidentifiable_nodes:
    #     unidentifiable_colors = [get_node_color(n, 'dark') for n in unidentifiable_nodes]
    #     nx.draw_networkx_nodes(G, pos, nodelist=unidentifiable_nodes, 
    #                           node_color=unidentifiable_colors, node_size=1000, 
    #                           edgecolors='red', linewidths=2, alpha=0.8)
    # Draw identifiable nodes (if any) - use same colors as unidentifiable for fill
    if identifiable_nodes:
        identifiable_colors = [get_node_color(n, 'light') for n in identifiable_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=identifiable_nodes, 
                            node_color=identifiable_colors, node_size=1000, 
                            edgecolors='green', linewidths=3, alpha=0.8)

    # Draw unidentifiable nodes - use same light colors for fill to match identifiable
    if unidentifiable_nodes:
        unidentifiable_colors = [get_node_color(n, 'light') for n in unidentifiable_nodes]  # Changed from 'dark' to 'light'
        nx.draw_networkx_nodes(G, pos, nodelist=unidentifiable_nodes, 
                            node_color=unidentifiable_colors, node_size=1000, 
                            edgecolors='red', linewidths=2, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2)
    
    # # CREATE EXTERNAL LABEL POSITIONS
    # label_pos = {}
    # for node, (x, y) in pos.items():
    #     # Calculate offset based on node position to avoid overlaps
    #     offset_x = 0.15 if x >= 0 else -0.15  # Push right if on right side, left if on left
    #     offset_y = 0.15 if y >= 0 else -0.15  # Push up if on top, down if on bottom
    #     label_pos[node] = (x + offset_x, y + offset_y)
    
    # # Draw labels at external positions
    # nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', 
    #                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    # CREATE EXTERNAL LABEL POSITIONS - closer to nodes
    label_pos = {}
    for node, (x, y) in pos.items():
        # Calculate smaller offset - closer to nodes
        offset_x = 0.04 if x >= 0 else -0.04  # Reduced from 0.15
        offset_y = 0.04 if y >= 0 else -0.04  # Reduced from 0.15
        label_pos[node] = (x + offset_x, y + offset_y)

    # Draw labels at external positions
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', 
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='States', edgecolor='black'),
        Patch(facecolor='lightcoral', label='Parameters', edgecolor='black'), 
        Patch(facecolor='lightgreen', label='Inputs', edgecolor='black'),
        Patch(facecolor='white', edgecolor='green', linewidth=3, label='Identifiable/\nObservable'),
        Patch(facecolor='white', edgecolor='red', linewidth=2, label='Unidentifiable/\nUnobservable')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Update title to include Observable/Unobservable
    updated_title = title.replace("Identifiability", "Identifiability/Observability")
    plt.title(updated_title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        # Save with larger margins to prevent label cutoff
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=1)
        print(f"Graph saved to: {save_path}")
        
        # Also save as vector format
        svg_path = save_path.with_suffix('.svg')
        plt.savefig(svg_path, dpi=300, bbox_inches='tight', pad_inches=1, format='svg')
        print(f"Graph saved to: {svg_path}")

    plt.close()


# def build_identifiability_graph(used_symbols, nullspace_vectors, symbol_types=None, index_range=None):
#     """
#     Construct a graph where nodes are variables (states/parameters/inputs).
#     Edges connect variables that are part of the same unidentifiable combination.
#     Isolated nodes are only identifiable if they don't appear in ANY nullspace vector.
    
#     Parameters
#     ----------
#     used_symbols : list of sympy.Symbol
#         Variables in the model.
#     nullspace_vectors : list 
#         Each vector encodes one unidentifiable direction.
#     symbol_types : list of str, optional
#         Type of each symbol ('state', 'param', 'input') for coloring
#     index_range : tuple (start, end) or None
#         Only look at indices [start:end] in the nullspace vectors
#         If None, use the full vector
#     """
#     G = nx.Graph()
    
#     # Add all nodes
#     for i, s in enumerate(used_symbols):
#         node_type = symbol_types[i] if symbol_types else 'unknown'
#         G.add_node(str(s), node_type=node_type)

#     variables_in_nullspace = set()
    
#     for vec in nullspace_vectors:
#         if index_range is None:
#             # Use full vector - symbol index i maps to vector index i
#             involved = [i for i in range(min(len(used_symbols), len(vec))) if vec[i] != 0] 
#         else:
#             # Use specified range - vector index (start_idx + i) maps to symbol index i
#             start_idx, end_idx = index_range
#             involved = []
#             for i in range(len(used_symbols)):
#                 vec_idx = start_idx + i
#                 if vec_idx < end_idx and vec_idx < len(vec) and vec[vec_idx] != 0:
#                     involved.append(i)  # i is the symbol index
        
#         # Mark all involved variables as unidentifiable
#         for symbol_idx in involved:
#             if symbol_idx < len(used_symbols):
#                 variables_in_nullspace.add(str(used_symbols[symbol_idx]))
        
#         # Add edges between all pairs in this nullspace vector
#         for i in range(len(involved)):
#             for j in range(i + 1, len(involved)):
#                 s1 = str(used_symbols[involved[i]])
#                 s2 = str(used_symbols[involved[j]])
#                 G.add_edge(s1, s2)

#     # Mark nodes as identifiable or unidentifiable
#     for node in G.nodes():
#         if node in variables_in_nullspace:
#             G.nodes[node]['identifiable'] = False
#         else:
#             G.nodes[node]['identifiable'] = True

#     return G


# def visualize_identifiability_graph(G, title="Variable Identifiability Graph", save_path=None):
#     """
#     Visualize the identifiability graph with nodes colored by variable type and identifiability.
#     """
#     plt.figure(figsize=(12, 9))
#     pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

#     # Separate nodes by identifiability
#     identifiable_nodes = [n for n in G.nodes() if G.nodes[n].get('identifiable', False)]
#     unidentifiable_nodes = [n for n in G.nodes() if not G.nodes[n].get('identifiable', True)]
    
#     # Color by variable type
#     def get_node_color(node, base_color):
#         node_type = G.nodes[node].get('node_type', 'unknown')
#         if node_type == 'state':
#             return 'lightblue' if base_color == 'light' else 'darkblue'
#         elif node_type == 'param':
#             return 'lightcoral' if base_color == 'light' else 'darkred'
#         elif node_type == 'input':
#             return 'lightgreen' if base_color == 'light' else 'darkgreen'
#         else:
#             return 'lightgray' if base_color == 'light' else 'darkgray'

#     # Draw identifiable nodes
#     if identifiable_nodes:
#         identifiable_colors = [get_node_color(n, 'light') for n in identifiable_nodes]
#         nx.draw_networkx_nodes(G, pos, nodelist=identifiable_nodes, 
#                             node_color=identifiable_colors, node_size=1000, 
#                             edgecolors='green', linewidths=3, alpha=0.8)

#     # Draw unidentifiable nodes
#     if unidentifiable_nodes:
#         unidentifiable_colors = [get_node_color(n, 'light') for n in unidentifiable_nodes]
#         nx.draw_networkx_nodes(G, pos, nodelist=unidentifiable_nodes, 
#                             node_color=unidentifiable_colors, node_size=1000, 
#                             edgecolors='red', linewidths=2, alpha=0.8)

#     # Draw edges
#     nx.draw_networkx_edges(G, pos, alpha=0.6, width=2)
    
#     # Create external label positions
#     label_pos = {}
#     for node, (x, y) in pos.items():
#         offset_x = 0.04 if x >= 0 else -0.04
#         offset_y = 0.04 if y >= 0 else -0.04
#         label_pos[node] = (x + offset_x, y + offset_y)

#     # Draw labels
#     nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold', 
#                         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
#     # Create legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='lightblue', label='States', edgecolor='black'),
#         Patch(facecolor='lightcoral', label='Parameters', edgecolor='black'), 
#         Patch(facecolor='lightgreen', label='Inputs', edgecolor='black'),
#         Patch(facecolor='white', edgecolor='green', linewidth=3, label='Identifiable/\nObservable'),
#         Patch(facecolor='white', edgecolor='red', linewidth=2, label='Unidentifiable/\nUnobservable')
#     ]
#     plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

#     updated_title = title.replace("Identifiability", "Identifiability/Observability")
#     plt.title(updated_title, fontsize=14, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=1)
#         print(f"Graph saved to: {save_path}")
        
#         # Also save as vector format
#         svg_path = save_path.with_suffix('.svg')
#         plt.savefig(svg_path, dpi=300, bbox_inches='tight', pad_inches=1, format='svg')
#         print(f"Graph saved to: {svg_path}")

#     plt.close()