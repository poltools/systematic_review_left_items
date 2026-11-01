import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from datetime import datetime
import matplotlib.patheffects as pe

def build_df_pairs(
    df: pd.DataFrame,
    year_column: str,
    category_column: str,
    embedding_column: str,
    years: list[int] = None
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [Year, Cat1, Cat2, Overlap], where Overlap
    is the mean cosine similarity between all embeddings in Cat1 vs Cat2
    among items with Year <= cutoff.

    Now filters out any embeddings whose length != the most common length,
    avoiding empt array or shape mismatches in vstack.
    """
    # --- 0) Year numeric & drop invalids ---
    df = df.copy()
    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
    df = df.dropna(subset=[year_column]).reset_index(drop=True)

    # --- 1) Keep only proper 1D np.ndarray embeddings ---
    mask_arr = df[embedding_column].apply(lambda e: isinstance(e, np.ndarray) and e.ndim == 1)
    df = df.loc[mask_arr].reset_index(drop=True)

    # --- 2) Filter to most common embedding length ---
    lengths = df[embedding_column].apply(lambda e: e.shape[0])
    common_len = lengths.value_counts().idxmax()
    df = df.loc[lengths == common_len].reset_index(drop=True)

    # --- 3) Determine which cutoff years to use ---
    if years is None:
        years = sorted(df[year_column].unique())

    # --- 4) List of all categories ---
    cats = df[category_column].dropna().unique().tolist()

    records = []
    for yr in years:
        # 5) subset up to this cutoff
        sub = df[df[year_column] <= yr]

        # 6) group embeddings by category → list of arrays
        grp = sub.groupby(category_column)[embedding_column].apply(list)

        # 7) For each unordered distinct pair, compute mean cosine sim
        for cat1, cat2 in combinations(cats, 2):
            if cat1 in grp.index and cat2 in grp.index:
                embs1 = np.vstack(grp[cat1])
                embs2 = np.vstack(grp[cat2])
                sim   = cosine_similarity(embs1, embs2).mean()
                records.append({
                    'Year':    yr,
                    'Cat1':    cat1,
                    'Cat2':    cat2,
                    'Overlap': sim
                })

        # 8) (Optional) self‐overlap
        for cat in cats:
            if cat in grp.index:
                records.append({
                    'Year':    yr,
                    'Cat1':    cat,
                    'Cat2':    cat,
                    'Overlap': 1.0
                })

    return pd.DataFrame.from_records(records)



def plot_category_network_3d(
    df_pairs: pd.DataFrame,
    year: int | None = None,
    cat1_col: str = 'Cat1',
    cat2_col: str = 'Cat2',
    weight_col: str = 'Overlap',
    top_n_edges: int | None = None,
    category_palette: dict[str, str] = None,
    dmin: float = 0.5,
    dmax: float = 3.0,
    gamma: float | None = None,
    Z0: float = 8.0,
    palette_name: str = 'tab10',
    cmap_name: str = 'Spectral',
    edge_width: float = 1.5,
    grayscale_edges: bool = False,
    show_labels: bool = True,
    figsize: tuple = (10, 8),
    dpi: int = 200
):
    """
    3D category similarity network:
      - Exact edge lengths via 3D MDS on remapped similarities
      - Perspective projection onto 2D plane
      - Depth-sorted, shaded edges colored by weight, or
        in uniform gray with width & darkness scaling when grayscale_edges=True
      - Colored nodes from custom or chosen palette; crisp labels; no frame
    """
    # 1) Build graph
    df = df_pairs.copy()
    if year is not None:
        df = df[df['Year'] == year]
    df = df[df[cat1_col] != df[cat2_col]]
    if top_n_edges is not None and not df.empty:
        df = df.nlargest(top_n_edges, weight_col)
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r[cat1_col], r[cat2_col], raw_weight=r[weight_col])

    # 2) Normalize & remap similarity → distance
    ws = np.array([d['raw_weight'] for *_, d in G.edges(data=True)])
    wmin, wmax = (ws.min(), ws.max()) if ws.size else (0.0, 1.0)
    for u, v, d in G.edges(data=True):
        norm = (d['raw_weight'] - wmin) / (wmax - wmin) if wmax > wmin else 0.5
        if gamma is not None:
            norm = norm ** gamma
        d['distance'] = dmax - norm * (dmax - dmin)
        d['w_norm'] = norm

    # 3) 3D MDS for exact pairwise distances
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n), float)
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        D[i, j] = D[j, i] = d['distance']
    coords3 = MDS(n_components=3, dissimilarity='precomputed', random_state=0)
    coords3 = coords3.fit_transform(D)

    # 4) Perspective projection onto z=0
    proj = lambda x, y, z: (x / (1 + z / Z0), y / (1 + z / Z0))
    pos2d = {nodes[i]: proj(*coords3[i]) for i in range(n)}

    # 5) Depth ordering
    zs_all = coords3[:, 2]
    min_z, max_z = zs_all.min(), zs_all.max()
    range_z = max_z - min_z if max_z > min_z else 1.0
    edge_list = [(u, v, (coords3[idx[u], 2] + coords3[idx[v], 2]) / 2)
                 for u, v in G.edges()]
    edge_list.sort(key=lambda x: x[2], reverse=True)

    # 6) Node colors & sizes
    if category_palette:
        node_colors = [category_palette.get(n, '#888888') for n in nodes]
    else:
        pal = sns.color_palette(palette_name, n_colors=n)
        node_colors = [pal[i] for i in range(n)]
    deg = dict(G.degree()); max_deg = max(deg.values()) if deg else 1
    node_sizes = [300 + 800 * (deg[nn] / max_deg) for nn in nodes]

    # 7) Plot setup
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    ax.axis('off'); ax.set_frame_on(False)

    # 8) Draw edges back-to-front
    for u, v, z in edge_list:
        x0, y0 = pos2d[u]; x1, y1 = pos2d[v]
        z_norm = (z - min_z) / range_z
        alpha = 0.1 + 0.6 * (1 - z_norm)
        w_norm = G[u][v]['w_norm']
        if grayscale_edges:
            # shade: darker for higher weight (w_norm=1 -> shade_min)
            shade_min, shade_max = 0.3, 0.8
            shade = shade_max - (shade_max - shade_min) * w_norm
            color = (shade, shade, shade)
            lw = edge_width * (1 + 2 * w_norm)  # width scales from 1× to 3×
            ax.plot([x0, x1], [y0, y1], color=color,
                    linewidth=lw, alpha=alpha, zorder=1)
        else:
            cmap = plt.get_cmap(cmap_name)
            color = cmap(w_norm)
            ax.plot([x0, x1], [y0, y1], color=color,
                    linewidth=edge_width, alpha=alpha, zorder=1)

    # 9) Draw nodes
    xs, ys = zip(*[pos2d[nn] for nn in nodes])
    ax.scatter(xs, ys, s=node_sizes, c=node_colors,
               edgecolors='white', linewidths=1.5, zorder=2)

    # 10) Labels
    if show_labels:
        for nn in nodes:
            x, y = pos2d[nn]
            ax.text(x, y, nn, fontsize=12, fontweight='bold', ha='center', va='center',
                    color='black', path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                    zorder=3)

    # 11) Colorbar (only when not grayscale)
    if not grayscale_edges:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name),
                                   norm=plt.Normalize(vmin=wmin, vmax=wmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([wmin, wmax])
        cbar.set_ticklabels(['Lower similarity', 'Higher similarity'])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig.savefig(f"figures/{timestamp}.png", dpi=300, bbox_inches='tight')

    plt.show()




def plot_category_similarity_heatmap_from_pairs(
    df_pairs: pd.DataFrame,
    year: int | None = None,
    cat1_col: str = "Cat1",
    cat2_col: str = "Cat2",
    sim_col: str = "Overlap",
    order: list[str] | None = None,
    palette: dict[str, str] | None = None,
    cmap: str = "viridis",
    figsize: tuple = (6.4, 4.8),
    linewidth: float = 0.5,
    save_path: str | None = None,
):
    df = df_pairs.copy()

    # Single snapshot (year) or mean across years
    if year is not None:
        df = df[df["Year"] == year]
        if df.empty:
            raise ValueError(f"No pairs for Year={year}.")
    else:
        df = df.groupby([cat1_col, cat2_col], as_index=False)[sim_col].mean()

    # Symmetrize: add swapped pairs
    df_swapped = df.rename(columns={cat1_col: cat2_col, cat2_col: cat1_col})
    df_sym = pd.concat([df, df_swapped], ignore_index=True)

    # Add diagonal = 1 for any category that appears at least once
    cats = pd.unique(df_sym[[cat1_col, cat2_col]].values.ravel("K"))
    diag = pd.DataFrame({cat1_col: cats, cat2_col: cats, sim_col: 1.0})
    df_sym = pd.concat([df_sym, diag], ignore_index=True)

    # Pivot to square matrix
    M = df_sym.pivot_table(index=cat1_col, columns=cat2_col, values=sim_col, aggfunc="mean")

    # (Optional) enforce order; otherwise keep matrix order
    if order is not None:
        # keep only categories that actually exist after filtering
        order_present = [c for c in order if c in M.index]
        if not order_present:
            raise ValueError("None of the requested categories are present after filtering.")
        M = M.reindex(index=order_present, columns=order_present)
    else:
        # drop any categories with all-NaN rows/cols (no data)
        good = (~M.isna()).any(axis=1) | (~M.isna()).any(axis=0)
        M = M.loc[good, good]

    # Replace diagonal with NaN so it appears white in the plot
    np.fill_diagonal(M.values, np.nan)

    # Determine color scale from off-diagonal values only
    vals = M.values[~np.isnan(M.values)]
    if vals.size == 0:
        raise ValueError("No pairwise similarities available to plot (all off-diagonal values are NaN).")
    vmin, vmax = float(vals.min()), float(vals.max())

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        M, cmap=cmap, square=True,
        vmin=vmin, vmax=vmax,
        cbar_kws={"label": "Cosine similarity"},
        linewidths=linewidth, linecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Color tick labels with your palette (if provided)
    if palette:
        for lab in ax.get_xticklabels():
            txt = lab.get_text()
            if txt in palette:
                lab.set_color(palette[txt])
                lab.set_rotation(45)
                lab.set_ha("right")
        for lab in ax.get_yticklabels():
            txt = lab.get_text()
            if txt in palette:
                lab.set_color(palette[txt])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
