import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from adjustText import adjust_text
import textwrap
import math

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def plot_category_evolution(
    df: pd.DataFrame,
    year_column: str,
    embedding_column: str,
    text_column: str,
    category_column: str,
    cutoff_years: list[int],
    mode: str = 'cumulative',       # 'cumulative' or 'snapshot'
    window: int = 5,                # half-window for snapshot
    # clustering pipeline params
    prep_reducer: str     = 'pca',  # 'pca','umap','tsne'
    prep_kwargs: dict     = None,
    pca_components: int   = 50,
    cluster_method: str   = 'kmeans',  # 'kmeans' or 'dbscan'
    cluster_kwargs: dict  = None,
    plot_elbow: bool      = False,
    elbow_range: tuple[int,int] = (1,10),
    viz_reducer: str      = 'tsne',   # 'pca','umap','tsne'
    viz_kwargs: dict      = None,
    random_state: int     = 42,
    palette_name: str     = 'tab20c',
    cols: int             = 3,
    figsize_per_plot: tuple = (5,5),
    point_alpha: float    = 0.03,
    centroid_alpha: float = 1.0,
    centroid_size: int    = 60,
    k_nearest: int        = 5,
    wrap_width: int       = 30,
    file_title: str       = "figure",
    show_labels: bool     = True
) -> pd.DataFrame:
    """
    Plot category medoid trajectories (cumulative or snapshot) on a 2D embedding.
    """
    # --- 0) Year numeric ---
    df = df.copy()
    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
    df = df.dropna(subset=[year_column]).reset_index(drop=True)

    # --- 1) Gather embeddings of consistent length ---
    df0 = df[df[embedding_column].apply(lambda e: isinstance(e, np.ndarray))].reset_index(drop=True)
    lengths = df0[embedding_column].apply(lambda e: e.shape[0] if e.ndim==1 else -1)
    common = lengths.value_counts().idxmax()
    df0 = df0[
        df0[embedding_column].apply(lambda e: isinstance(e,np.ndarray) and e.ndim==1 and e.shape[0]==common)
    ].reset_index(drop=True)
    X = np.stack(df0[embedding_column].values)

    # --- 2) Pre‐clustering reduction ---
    pk = prep_kwargs or {}
    pr = prep_reducer.lower()
    if pr == 'pca':
        X_pre = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)
    elif pr == 'umap' and _HAS_UMAP:
        X_pre = UMAP(n_components=pca_components, random_state=random_state, **pk).fit_transform(X)
    elif pr == 'tsne':
        X_pre = TSNE(n_components=pca_components, random_state=random_state, **pk).fit_transform(X)
    else:
        X_pre = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)

    # --- 3) Optional elbow ---
    if cluster_method.lower()=='kmeans' and plot_elbow:
        ck0 = (cluster_kwargs or {}).copy()
        ck0.pop('n_clusters', None)
        ks = list(range(elbow_range[0], elbow_range[1]+1))
        inertias = [KMeans(n_clusters=k, random_state=random_state, **ck0).fit(X_pre).inertia_ for k in ks]
        plt.figure(figsize=(6,4))
        plt.plot(ks, inertias, '-o')
        plt.title("K-Means Elbow Test")
        plt.xlabel("n_clusters"); plt.ylabel("inertia")
        plt.xticks(ks); plt.show()

    # --- 4) Clustering ---
    ck = cluster_kwargs or {}
    cmeth = cluster_method.lower()
    if cmeth == 'kmeans':
        labels = KMeans(random_state=random_state, **ck).fit_predict(X_pre)
    elif cmeth == 'dbscan':
        labels = DBSCAN(**ck).fit_predict(X_pre)
    else:
        raise ValueError("cluster_method must be 'kmeans' or 'dbscan'")
    df0['cluster'] = labels

    # --- 5) 2D embedding for display ---
    vk = viz_kwargs or {}
    vr = viz_reducer.lower()
    if vr == 'pca':
        vis = PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif vr == 'umap' and _HAS_UMAP:
        vis = UMAP(n_components=2, random_state=random_state, **vk).fit_transform(X)
    elif vr == 'tsne':
        vis = TSNE(n_components=2, random_state=random_state, **vk).fit_transform(X)
    else:
        vis = PCA(n_components=2, random_state=random_state).fit_transform(X)
    df_plot = df0.copy()
    df_plot['dim1'], df_plot['dim2'] = vis[:,0], vis[:,1]

    # --- 6) Palette (tab20c + override last two) ---
    cats = sorted(df_plot[category_column].unique())
    if palette_name.lower()=='tab20c':
        base = cm.get_cmap('tab20c', len(cats))
        cols_list = [base(i) for i in range(len(cats))]
        cols_list[-2] = '#d62728'
        cols_list[-1] = '#ff7f0e'
    else:
        cmap = cm.get_cmap(palette_name, len(cats))
        cols_list = list(getattr(cmap, 'colors', [cmap(i) for i in range(len(cats))]))
    colmap = dict(zip(cats, cols_list))

    # --- 7) Compute medoids & label texts ---
    medoids = {}
    exemplars = {}
    for year in cutoff_years:
        if mode=='cumulative':
            mask = df_plot[year_column] <= year
        else:
            mask = df_plot[year_column].between(year-window, year+window)
        sub = df_plot[mask]
        for cat in cats:
            pts = sub[sub[category_column]==cat]
            if pts.empty:
                medoids[(cat,year)] = (np.nan,np.nan)
                exemplars[(cat,year)] = ""
            else:
                coords = pts[['dim1','dim2']].values
                D = pairwise_distances(coords)
                idx = np.argmin(D.sum(axis=1))
                mx,my = coords[idx]
                medoids[(cat,year)] = (mx, my)
                if show_labels:
                    text_pt = pts.iloc[idx][text_column]
                    exemplars[(cat,year)] = textwrap.fill(str(text_pt), width=wrap_width)

    # --- 8) Plot grid ---
    n = len(cutoff_years)
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_plot[0]*cols,
                                      figsize_per_plot[1]*rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # --- 9) Draw each panel ---
    for ax, year in zip(axes, cutoff_years):
        title = f"Up through {year}" if mode=='cumulative' else f"{year}±{window}"
        if mode=='cumulative':
            mask = df_plot[year_column] <= year
        else:
            mask = df_plot[year_column].between(year-window, year+window)

        # 9a) background cloud
        ax.scatter(df_plot.loc[mask,'dim1'], df_plot.loc[mask,'dim2'],
                   color='gray', alpha=point_alpha,
                   s=10, edgecolor='none')

        # 9b) if cumulative: draw medoid trajectories
        if mode=='cumulative':
            for cat in cats:
                xs, ys = [], []
                for yr in cutoff_years:
                    if yr>year: break
                    mx,my = medoids[(cat,yr)]
                    if not np.isnan(mx):
                        xs.append(mx); ys.append(my)
                ax.plot(xs, ys, '-', color=colmap[cat], alpha=centroid_alpha)

        # 9c) draw medoid points + labels
        texts = []
        for cat in cats:
            mx,my = medoids[(cat,year)]
            if not np.isnan(mx):
                ax.scatter(mx, my,
                           color=colmap[cat],
                           s=centroid_size,
                           alpha=centroid_alpha,
                           edgecolor='k', linewidth=0.5)
                if show_labels:
                    texts.append(ax.text(mx, my, exemplars[(cat,year)],
                                         fontsize=7, weight='bold',
                                         color=colmap[cat]))
        if show_labels and texts:
            adjust_text(
                texts, ax=ax,
                expand_text=(3,3), expand_points=(3,3),
                force_text=(2,2), force_points=(2,2),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                lim=2000
            )

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    # --- 10) Remove extra axes ---
    for ax in axes[n:]:
        ax.remove()

    # --- 11) Legend ---
    handles = [
        plt.Line2D([],[],marker='o', ls='', color=colmap[c], label=c)
        for c in cats
    ]
    fig.legend(handles, cats,
               loc='lower center', ncol=len(cats), frameon=False)

    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(f"{file_title}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return df_plot