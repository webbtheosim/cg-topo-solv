import numpy as np
import networkx as nx
import proplot as pplt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

cmap = pplt.Colormap('coolwarm')
norm = mcolors.Normalize(vmin=0.0, vmax=0.5)

def plot_graph(G, ax, label, iteration=1000, node_size=10, lw=0.5, seed=42):
    """Plot a graph with PCA-aligned node positions and colored by label."""
    
    np.random.seed(seed)
    pos = nx.kamada_kawai_layout(G)
    pos = nx.spring_layout(G, pos=pos, iterations=iteration)

    node_ids = list(G.nodes)
    coords = np.array([pos[n] for n in node_ids])
    coords_pca = PCA(n_components=2).fit_transform(coords)
    pos_rotated = {node_ids[i]: coords_pca[i] for i in range(len(node_ids))}

    node_color = cmap(norm(label))

    nx.draw_networkx_nodes(
        G,
        pos=pos_rotated,
        node_color=[node_color] * len(G.nodes),
        node_size=node_size,
        edgecolors="black",
        linewidths=lw,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos=pos_rotated, edge_color="k", ax=ax)

    ax.set_aspect("equal")
    ax.set_xlim(coords_pca[:, 0].min() - 0.8, coords_pca[:, 0].max() + 0.8)
    ax.set_ylim(coords_pca[:, 1].min() - 0.8, coords_pca[:, 1].max() + 0.8)
    ax.axis("off")

    return ax
