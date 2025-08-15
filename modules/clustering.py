import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional deps
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

try:
    from umap import UMAP
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


# ---------- helpers ----------

def _safe_pca_n_components(X: np.ndarray, requested: int) -> int:
    """Clamp PCA components to a valid range."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return 1
    return max(1, min(requested, n_features, n_samples - 1))


def _maybe_clamp(value, lo=None, hi=None):
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _effective_tsne_perplexity(n_samples: int, kwargs: dict) -> float:
    """t-SNE requires 1 < perplexity < n_samples; commonly < (n_samples-1)/3."""
    perpl = kwargs.get("perplexity", 30)
    upper = max(2, (n_samples - 1) // 3)
    return _maybe_clamp(perpl, lo=2, hi=upper)


def _tsne_fit_transform(X: np.ndarray, n_components: int, random_state: int, kwargs: dict | None):
    """Version-proof TSNE call: filter/rename kwargs and auto-fix incompatibilities."""
    import inspect

    kwargs = dict(kwargs or {})
    sig = inspect.signature(TSNE.__init__)

    # Keep only supported kwargs
    tsne_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Remap n_iter -> max_iter if needed
    if "n_iter" in kwargs and "n_iter" not in sig.parameters and "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = kwargs["n_iter"]

    # Perplexity must be valid for current sample size
    n_samples = X.shape[0]
    if "perplexity" in kwargs:
        tsne_kwargs["perplexity"] = _effective_tsne_perplexity(n_samples, kwargs)

    # Barnes-Hut supports n_components <= 3; force exact if higher
    if n_components > 3 and "method" in sig.parameters:
        tsne_kwargs.setdefault("method", "exact")

    return TSNE(n_components=n_components, random_state=random_state, **tsne_kwargs).fit_transform(X)


# ---------- defaults for category colors (for labeled category plots) ----------

DEFAULT_CATEGORY_PALETTE = {
    'communism': '#1f77b4',                  # blue
    'egalitarianism': '#ff7f0e',             # orange
    'environmentalism': '#2ca02c',           # green
    'feminism': '#9467bd',                   # purple
    'left-wing authoritarianism': '#7f7f7f', # dark grey
    'progressivism': '#c7c7c7',              # light grey
}

DEFAULT_LEGEND_ORDER = [
    'communism',
    'egalitarianism',
    'environmentalism',
    'feminism',
    'left-wing authoritarianism',
    'progressivism',
]


# ---------- main API ----------

def plot_cumulative_embedding_pipeline(
    df: pd.DataFrame,
    embedding_col: str = None,
    token_col: str | None = None,
    use_tfidf: bool = False,
    category_col: str | None = None,
    years_to_plot: list[int] | None = None,
    # Pre-clustering reduction
    prep_reducer: str     = 'pca',          # 'pca' | 'umap' | 'tsne'
    prep_kwargs: dict     = None,
    pca_components: int   = 50,
    # Clustering
    cluster_method: str   = 'kmeans',       # 'kmeans' | 'dbscan' | 'hdbscan'
    cluster_kwargs: dict  = None,
    plot_elbow: bool      = False,
    elbow_range: tuple[int, int] = (1, 10),
    # Visualization reduction
    viz_reducer: str      = 'umap',         # 'umap' | 'tsne' | 'pca'
    viz_kwargs: dict      = None,
    # General
    random_state: int     = 42,
    palette_name: str     = 'tab20',        # used for cluster coloring when category_col is None
    cols: int             = 3,
    figsize_per_plot: tuple = (5, 5),
    legend: bool          = True,
    wider_legend: bool    = False,          # False → legend on the RIGHT; True → legend at the BOTTOM
    title: str | None     = None,
    # Category plot styling
    category_palette: dict | None = None,   # mapping category -> color (if category_col is provided)
    legend_order: list[str] | None = None,  # order for categories in legend
    # Point styling
    point_size: int = 18,
    point_alpha: float = 0.7,
    # NEW: borders around points
    point_edgecolor: str | None = 'white',  # None disables border
    point_edgewidth: float = 0.6,
    outline_mode: str = 'edge',             # 'edge' (single scatter) or 'halo' (double scatter)
) -> pd.DataFrame:
    """
    Full pipeline for embedding analysis:
      - Optionally compute TF-IDF from a token column.
      - Dimensionality reduction for clustering.
      - Clustering (KMeans, DBSCAN, HDBSCAN).
      - 2D visualization reduction.
      - Plot cumulative embeddings by year.
      - Returns annotated DataFrame.

    Legend behavior:
      - legend=True, wider_legend=False → legend stacked on the RIGHT.
      - legend=True, wider_legend=True  → legend centered at the BOTTOM across columns.

    Point borders:
      - outline_mode='edge' uses Matplotlib's edgecolors/linewidths (fast).
      - outline_mode='halo' draws a slightly larger solid dot underneath so the
        outline isn't faded by alpha (crisper but draws twice).
    """
    if years_to_plot is None:
        years_to_plot = []

    df2 = df.copy().reset_index(drop=True)

    # ---- Year handling
    if 'Year' not in df2.columns:
        warnings.warn("Column 'Year' not found; setting to NaN.")
        df2['Year'] = np.nan
    df2['Year'] = pd.to_numeric(df2['Year'], errors='coerce').astype('Int64')

    # ---- Build feature matrix X
    if use_tfidf:
        assert token_col is not None, "You must specify `token_col` when `use_tfidf` is True"
        assert all(isinstance(x, list) for x in df2[token_col]), "`token_col` must contain lists of tokens"
        vectorizer = TfidfVectorizer(analyzer=lambda x: x)
        X = vectorizer.fit_transform(df2[token_col]).toarray()
    else:
        if embedding_col is None:
            raise ValueError("embedding_col must be provided when use_tfidf=False")
        # Coerce to arrays and filter good rows
        def _to_arr(x):
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, (list, tuple)):
                try:
                    return np.asarray(x, dtype=float)
                except Exception:
                    return None
            return None

        arrs = df2[embedding_col].apply(_to_arr)
        ok = arrs.apply(lambda a: isinstance(a, np.ndarray) and a.ndim == 1 and a.size > 0 and np.all(np.isfinite(a)))
        df2 = df2.loc[ok].reset_index(drop=True)
        if df2.empty:
            raise ValueError("No valid embeddings after filtering.")
        lengths = df2[embedding_col].apply(lambda x: (x if isinstance(x, np.ndarray) else np.asarray(x)).size)
        expected = int(lengths.value_counts().idxmax())
        df2 = df2.loc[lengths == expected].reset_index(drop=True)
        if df2.empty:
            raise ValueError("All embeddings had inconsistent dimensionality.")
        X = np.stack(df2[embedding_col].values)

    n_samples = X.shape[0]

    # ---- Pre-clustering reduction
    pr = (prep_reducer or 'pca').lower()
    pk = dict(prep_kwargs or {})

    if pr == 'pca':
        n_comp = _safe_pca_n_components(X, pca_components)
        if n_comp < pca_components:
            warnings.warn(f"PCA components clamped to {n_comp} (samples/features limit).")
        X_prep = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
    elif pr == 'umap' and _HAS_UMAP:
        # Clamp neighbors to <= n_samples - 1
        if 'n_neighbors' in pk:
            pk['n_neighbors'] = _maybe_clamp(pk['n_neighbors'], lo=2, hi=max(2, n_samples - 1))
        umap_args = {k: pk[k] for k in ('n_neighbors', 'min_dist', 'metric', 'init', 'spread', 'set_op_mix_ratio') if k in pk}
        n_comp = _maybe_clamp(pca_components, lo=2, hi=max(2, X.shape[1]))
        X_prep = UMAP(n_components=n_comp, random_state=random_state, **umap_args).fit_transform(X)
    elif pr == 'tsne':
        n_comp = _maybe_clamp(pca_components, lo=2, hi=min(3, X.shape[1]))  # >3 triggers 'exact' method; keep small
        X_prep = _tsne_fit_transform(X, n_comp, random_state, pk)
    else:
        # Fallback to PCA
        n_comp = _safe_pca_n_components(X, pca_components)
        X_prep = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)

    # ---- Optional elbow for KMeans
    cmeth = (cluster_method or 'kmeans').lower()
    ck = dict(cluster_kwargs or {})
    ck_no_n = dict(ck)
    ck_no_n.pop('n_clusters', None)

    if cmeth == 'kmeans' and plot_elbow:
        k_min, k_max = elbow_range
        ks = [k for k in range(k_min, k_max + 1) if k <= n_samples and k >= 1]
        if not ks:
            warnings.warn("Elbow plot skipped (no valid k values for current sample size).")
        else:
            inertias = []
            for k in ks:
                inertias.append(KMeans(n_clusters=k, random_state=random_state, **ck_no_n).fit(X_prep).inertia_)
            plt.figure(figsize=(6, 4))
            plt.plot(ks, inertias, marker='o')
            plt.title("K-Means Elbow Test")
            plt.xlabel("n_clusters")
            plt.ylabel("Inertia")
            plt.xticks(ks)
            plt.show()

    # ---- Clustering
    if cmeth == 'kmeans':
        n_clusters = ck.get('n_clusters', 8)
        n_clusters = _maybe_clamp(n_clusters, lo=1, hi=n_samples)
        if n_clusters != ck.get('n_clusters', 8):
            warnings.warn(f"Adjusted kmeans n_clusters to {n_clusters} based on sample size.")
        ck['n_clusters'] = n_clusters
        labels = KMeans(random_state=random_state, **ck).fit_predict(X_prep)
    elif cmeth == 'dbscan':
        labels = DBSCAN(**ck).fit_predict(X_prep)
    elif cmeth == 'hdbscan':
        if not _HAS_HDBSCAN:
            raise ImportError("hdbscan is not installed. `pip install hdbscan` to use this method.")
        if 'n_clusters' in ck:
            ck['min_cluster_size'] = ck.pop('n_clusters')
            warnings.warn("HDBSCAN: converted n_clusters -> min_cluster_size.")
        labels = hdbscan.HDBSCAN(**ck).fit_predict(X_prep)
    else:
        raise ValueError("cluster_method must be 'kmeans', 'dbscan', or 'hdbscan'")

    df2['cluster'] = labels

    # ---- Clustering metrics (only if valid)
    real_clusters = [l for l in np.unique(labels) if l >= 0]
    if len(real_clusters) > 1 and n_samples >= 3:
        try:
            print("Clustering metrics:")
            print(f" • Silhouette Score:        {silhouette_score(X_prep, labels):.3f}")
            print(f" • Calinski–Harabasz Score: {calinski_harabasz_score(X_prep, labels):.1f}")
            print(f" • Davies–Bouldin Score:    {davies_bouldin_score(X_prep, labels):.3f}")
        except Exception as e:
            warnings.warn(f"Skipping metrics due to: {e}")
    else:
        print("Only one cluster (or all noise); skipping metric calculations.")

    # ---- 2D visualization reduction
    vr = (viz_reducer or 'pca').lower()
    vk = dict(viz_kwargs or {})
    if vr == 'umap' and _HAS_UMAP:
        if 'n_neighbors' in vk:
            vk['n_neighbors'] = _maybe_clamp(vk['n_neighbors'], lo=2, hi=max(2, n_samples - 1))
        umap_args = {k: vk[k] for k in ('n_neighbors','min_dist','metric','init','spread','set_op_mix_ratio') if k in vk}
        proj2d = UMAP(n_components=2, random_state=random_state, **umap_args).fit_transform(X)
    elif vr == 'tsne':
        vk['perplexity'] = _effective_tsne_perplexity(n_samples, vk)
        proj2d = _tsne_fit_transform(X, 2, random_state, vk)
    else:
        proj2d = PCA(n_components=2, random_state=random_state).fit_transform(X)
    df2['dim1'], df2['dim2'] = proj2d[:, 0], proj2d[:, 1]

    # ---- Plot cumulative panels

    # category-based palette (if category_col provided) OR cluster palette (colormap)
    use_categories = bool(category_col and category_col in df2)
    if use_categories:
        pal = dict(DEFAULT_CATEGORY_PALETTE)
        if category_palette:
            pal.update(category_palette)
        seen = list(pd.Index(df2[category_col].dropna().unique()))
        for g in seen:
            pal.setdefault(g, '#c7c7c7')
        order = list(legend_order or DEFAULT_LEGEND_ORDER)
        order += [g for g in seen if g not in order]
        hue = category_col
        groups_for_plot = order
        color_map = {g: pal[g] for g in groups_for_plot if g in pal}
    else:
        uniq = sorted(df2['cluster'].unique())
        cmap = cm.get_cmap(palette_name, len(uniq) or 1)
        color_map = {g: cmap(i) for i, g in enumerate(uniq)}
        groups_for_plot = uniq
        hue = 'cluster'

    n = len(years_to_plot) if years_to_plot else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    panels = years_to_plot if years_to_plot else ["All years"]

    # consistent limits across panels
    xlim = (float(df2['dim1'].min()), float(df2['dim1'].max()))
    ylim = (float(df2['dim2'].min()), float(df2['dim2'].max()))

    for ax, yr in zip(axes, panels):
        if years_to_plot:
            mask_year = df2['Year'] <= yr
            title_text = title or f"Up through {yr}"
        else:
            mask_year = np.ones(len(df2), dtype=bool)
            title_text = title or "All data"

        for g in groups_for_plot:
            pts = df2.loc[mask_year & (df2[hue] == g)]
            if pts.empty:
                continue

            face = color_map.get(g, (0.6, 0.6, 0.6, 1.0))

            if point_edgecolor and outline_mode == 'halo':
                # Draw an underlay "halo" a bit larger (alpha=1) then draw the face on top.
                halo_scale = 1.35  # size multiplier for the halo layer
                ax.scatter(
                    pts['dim1'], pts['dim2'],
                    s=point_size * halo_scale, alpha=1.0,
                    linewidths=0, edgecolors='none',
                    color=point_edgecolor, zorder=1
                )
                ax.scatter(
                    pts['dim1'], pts['dim2'],
                    s=point_size, alpha=point_alpha,
                    linewidths=0, edgecolors='none',
                    color=face, zorder=2
                )

            else:
                # Single scatter with edgecolor/linewidth (fast, looks great)
                lw = point_edgewidth if point_edgecolor else 0
                ec = point_edgecolor if point_edgecolor else 'none'
                ax.scatter(
                    pts['dim1'], pts['dim2'],
                    s=point_size, alpha=point_alpha,
                    linewidths=lw, edgecolors=ec,
                    color=face
                )

        ax.set_title(title_text)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ('top', 'right', 'left', 'bottom'):
            ax.spines[spine].set_visible(False)

    # remove unused axes
    for ax in axes[len(panels):]:
        ax.remove()

    # ----- legend placement
    if legend and len(groups_for_plot) > 0:
        handles = [
            plt.Line2D([], [], marker='o', linestyle='', markersize=6,
                       markerfacecolor=color_map[g],
                       markeredgecolor=(point_edgecolor if point_edgecolor else 'none'),
                       markeredgewidth=(point_edgewidth if point_edgecolor else 0),
                       label=str(g))
            for g in groups_for_plot if g in color_map
        ]

        if wider_legend:
            # bottom-centered legend across columns
            fig.legend(handles, [str(g) for g in groups_for_plot if g in color_map],
                       loc='lower center', ncol=len(handles),
                       frameon=False, borderaxespad=0.5)
            plt.tight_layout(rect=[0, 0.06, 1, 1])
        else:
            # right-side stacked legend
            fig.subplots_adjust(right=0.80)
            axes[0].legend(handles, [str(g) for g in groups_for_plot if g in color_map],
                           loc='upper left', bbox_to_anchor=(1.02, 1.0),
                           borderaxespad=0, frameon=False, fontsize='small')
            plt.tight_layout(rect=[0, 0.00, 0.98, 1])
    else:
        plt.tight_layout(rect=[0, 0.00, 1, 1])

    # Save and show
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return df2



def plot_embedding_2d_3d(
    df: pd.DataFrame,
    embedding_col: str = None,
    token_col: str    = None,
    use_tfidf: bool   = False,
    prep_reducer: str    = 'pca',      # 'pca' or 'umap'
    prep_kwargs: dict    = None,
    pca_components: int  = 50,
    cluster_method: str  = 'kmeans',   # 'kmeans' or 'dbscan'
    cluster_kwargs: dict = None,
    viz_reducer: str     = 'tsne',     # 'pca', 'tsne', or 'umap'
    viz_kwargs: dict     = None,
    category_col: str    = None,
    palette_name: str    = 'Accent',
    random_state: int    = 42,
    elev: int            = 30,
    azim: int            = 45,
    figsize: tuple       = (12, 6),
    legend: bool         = True,
    outlier_pct: float   = 99.0,
    category_palette: dict[str, str] = None,
) -> pd.DataFrame:
    """
    2D + 3D side-by-side embedding & clustering.
    Shared X/Y scales (and Z for 3D).
    """
    df2 = df.copy().reset_index(drop=True)

    # 1) Build X
    if use_tfidf:
        assert token_col, "`token_col` required if use_tfidf=True"
        vec = TfidfVectorizer(analyzer=lambda x: x)
        X = vec.fit_transform(df2[token_col]).toarray()
    else:
        mask = df2[embedding_col].apply(lambda v: isinstance(v, np.ndarray) and v.ndim == 1)
        df2 = df2.loc[mask].reset_index(drop=True)
        lengths = df2[embedding_col].apply(lambda v: v.shape[0])
        common = int(lengths.value_counts().idxmax())
        df2 = df2.loc[lengths == common].reset_index(drop=True)
        X = np.stack(df2[embedding_col].values)

    # 2) Pre-clustering reduction
    pk = prep_kwargs or {}
    if prep_reducer == 'pca':
        X_pre = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)
    elif prep_reducer == 'umap' and _HAS_UMAP:
        X_pre = UMAP(n_components=pca_components, random_state=random_state, **pk).fit_transform(X)
    else:
        X_pre = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)

    # 3) Clustering
    ck = cluster_kwargs or {}
    if cluster_method == 'kmeans':
        labels = KMeans(random_state=random_state, **ck).fit_predict(X_pre)
    elif cluster_method == 'dbscan':
        labels = DBSCAN(**ck).fit_predict(X_pre)
    else:
        raise ValueError("cluster_method must be 'kmeans' or 'dbscan'")
    df2['cluster'] = labels

    # metrics
    if len(set(labels)) > 1:
        print("Silhouette:", silhouette_score(X_pre, labels))
        print("Calinski–Harabasz:", calinski_harabasz_score(X_pre, labels))
        print("Davies–Bouldin:", davies_bouldin_score(X_pre, labels))

    # 4) 3D coords
    vk = viz_kwargs or {}
    vr = viz_reducer.lower()
    if vr == 'pca':
        coords = PCA(n_components=3, random_state=random_state).fit_transform(X)
    elif vr == 'tsne':
        coords = TSNE(n_components=3, random_state=random_state, **vk).fit_transform(X)
    elif vr == 'umap' and _HAS_UMAP:
        coords = UMAP(n_components=3, random_state=random_state, **vk).fit_transform(X)
    else:
        raise ValueError("viz_reducer must be 'pca','tsne' or 'umap'")

    # 5) Outlier removal
    center = coords.mean(axis=0)
    dists  = np.linalg.norm(coords - center, axis=1)
    thresh = np.percentile(dists, outlier_pct)
    keep   = dists < thresh
    coords = coords[keep]
    df2    = df2.loc[keep].reset_index(drop=True)

    # assign dims
    df2['dim1'], df2['dim2'] = coords[:, 0], coords[:, 1]
    df2['x'], df2['y'], df2['z'] = coords[:, 0], coords[:, 1], coords[:, 2]

    # Compute global axis limits with padding
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    pad_z = (z_max - z_min) * 0.05
    xlims = (x_min - pad_x, x_max + pad_x)
    ylims = (y_min - pad_y, y_max + pad_y)
    zlims = (z_min - pad_z, z_max + pad_z)

    # 6) Colors & legend handles
    hue    = category_col if category_col in df2.columns else 'cluster'
    groups = sorted(df2[hue].dropna().unique())

    if category_palette:
        default_cmap = cm.get_cmap(palette_name, len(groups))
        color_map = {g: category_palette.get(g, default_cmap(i)) for i, g in enumerate(groups)}
    else:
        cmap = cm.get_cmap(palette_name, len(groups))
        color_map = {g: cmap(i) for i, g in enumerate(groups)}

    handles = [
        plt.Line2D([], [], marker='o', ls='', color=color_map[g], label=str(g))
        for g in groups
    ]

    # 7) Plot
    fig = plt.figure(figsize=figsize)

    # 2D panel
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(xlims); ax1.set_ylim(ylims)
    for g in groups:
        pts = df2[df2[hue] == g]
        ax1.scatter(pts['dim1'], pts['dim2'], s=20, c=[color_map[g]],
                    alpha=0.7, edgecolors='k', linewidths=0.2)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel("Dimension 1"); ax1.set_ylabel("Dimension 2")

    # 3D panel
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_xlim(xlims); ax2.set_ylim(ylims); ax2.set_zlim(zlims)
    for g in groups:
        pts = df2[df2[hue] == g]
        ax2.scatter(pts['x'], pts['y'], pts['z'], s=15, c=[color_map[g]],
                    alpha=0.7, edgecolors='k', linewidths=0.2)

    # ——— here come our fixes ———
    ax2.view_init(elev=elev, azim=azim)
    ax2.dist = 9  # smaller → zoom in, larger → zoom out
    ax2.tick_params(axis='x', rotation=0)
    ax2.tick_params(axis='y', rotation=0)
    ax2.tick_params(axis='z', rotation=0)
    ax2.set_xlabel("Dimension 1", labelpad=12)
    ax2.set_ylabel("Dimension 2", labelpad=12)
    ax2.set_zlabel("Dimension 3", labelpad=8)
    # ————————————————————————
    fig.subplots_adjust(wspace=0.9)   # default is ~0.2; increase to push them apart

    if legend:
        fig.legend(
            handles, [str(g) for g in groups],
            loc='lower center', ncol=max(1, len(groups)//2),
            frameon=False, bbox_to_anchor=(0.5, -0.05),
            bbox_transform=fig.transFigure
        )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return df2