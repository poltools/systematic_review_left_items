import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import warnings


def clean_embeddings_column(df, embedding_col):
    def to_array(x):
        if isinstance(x, np.ndarray): return x
        if isinstance(x, (list, tuple)):
            try: return np.asarray(x, dtype=float)
            except: return None
        return None

    arrs = df[embedding_col].apply(to_array)
    mask = arrs.apply(lambda v: isinstance(v, np.ndarray) and v.ndim==1 and v.size>0 and np.all(np.isfinite(v)))
    out = df.loc[mask].copy()
    out[embedding_col] = arrs.loc[mask]
    return out



def find_similar_groups(
    df: pd.DataFrame,
    embedding_col: str,
    content_col: str = 'Item content',
    n_neighbors: int = 5,
    similarity_threshold: float | None = None,
    metric: str = 'cosine'
) -> tuple[pd.DataFrame, pd.Series]:

    def to_array(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, (list, tuple)):
            try:
                return np.asarray(x, dtype=float)
            except Exception:
                return None
        return None

    # 0) coerce to ndarray where possible
    coerced = df[embedding_col].apply(to_array)

    # 1) filter usable rows
    is_array = coerced.apply(lambda x: isinstance(x, np.ndarray))
    n_not_array = (~is_array).sum()

    # drop zero-length, non-finite, and non-1D
    def ok_vec(v):
        if not isinstance(v, np.ndarray):
            return False
        if v.ndim != 1:
            return False
        if v.size == 0:
            return False
        if not np.all(np.isfinite(v)):
            return False
        return True

    ok_mask = coerced.apply(ok_vec)
    n_bad_shape_or_vals = (~ok_mask & is_array).sum()

    df_valid = df.loc[ok_mask].copy()
    df_valid[embedding_col] = coerced.loc[ok_mask]

    # 2) consistent dimensionality
    if df_valid.empty:
        raise ValueError(
            f"No valid embeddings after filtering. "
            f"Dropped {n_not_array} rows (not array/convertible) and "
            f"{n_bad_shape_or_vals} rows (empty/non-finite/non-1D)."
        )

    lengths = df_valid[embedding_col].apply(lambda x: x.size)
    expected_dim = lengths.value_counts().idxmax()
    correct = lengths == expected_dim
    if not correct.all():
        warnings.warn(f"Dropping {correct.size - correct.sum()} rows of size ≠ {expected_dim}")
        df_valid = df_valid.loc[correct]

    if df_valid.empty:
        raise ValueError("All embeddings had inconsistent dimensionality.")

    # 3) build X and index mapping
    idxs = df_valid.index.to_list()
    X = np.vstack(df_valid[embedding_col].values)

    # 4) k-NN (guard for tiny sets)
    k = min(n_neighbors + 1, max(1, len(df_valid)))
    if k <= 1:
        # Only one point — no neighbors; return empty edges and a single group
        neighbors_df = pd.DataFrame(columns=['index','content','neighbor_index','neighbor_content','similarity'])
        groups = pd.Series({idxs[0]: 0}, name='group').reindex(df.index)
        return neighbors_df, groups

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    dists, neighb = nn.kneighbors(X, return_distance=True)

    # 5) neighbor list
    records = []
    for row_pos, (src_idx, dist_row, nbr_row) in enumerate(zip(idxs, dists, neighb)):
        src_text = df_valid.at[src_idx, content_col]
        for dist, nbr_i in zip(dist_row[1:], nbr_row[1:]):
            nbr_idx = idxs[nbr_i]
            sim = 1 - dist if metric == 'cosine' else -dist
            nbr_text = df_valid.at[nbr_idx, content_col]
            records.append({
                'index':            src_idx,
                'content':          src_text,
                'neighbor_index':   nbr_idx,
                'neighbor_content': nbr_text,
                'similarity':       sim
            })

    neighbors_df = pd.DataFrame(records)

    # 6) optional pruning
    if similarity_threshold is not None and metric == 'cosine':
        before = len(neighbors_df)
        neighbors_df = neighbors_df[neighbors_df['similarity'] >= similarity_threshold]
        after = len(neighbors_df)
        warnings.warn(f"Pruned {before-after} edges below sim {similarity_threshold}")

    # 7) graph & components
    G = nx.Graph()
    G.add_nodes_from(idxs)
    for _, row in neighbors_df.iterrows():
        G.add_edge(row['index'], row['neighbor_index'])

    comp = nx.connected_components(G)
    group_map = {node: gid for gid, comp_nodes in enumerate(comp) for node in comp_nodes}

    groups = pd.Series(group_map, name='group').reindex(df.index)
    return neighbors_df.reset_index(drop=True), groups


def deduplicate_items(items_df, highly_similar):
    # 0) edges from your existing table
    pairs = (highly_similar[['index', 'neighbor_index']]
             .assign(u=lambda d: d[['index', 'neighbor_index']].min(axis=1),
                     v=lambda d: d[['index', 'neighbor_index']].max(axis=1))
             .loc[lambda d: d.u != d.v, ['u', 'v']]
             .drop_duplicates())

    # 1) connected components over those edges
    G = nx.Graph()
    G.add_edges_from(pairs.itertuples(index=False, name=None))
    components = list(nx.connected_components(G))

    # 2) canonical = smallest .loc index in each component
    rep_map = {i: min(comp) for comp in components for i in comp}
    duplicate_to_canonical = pd.Series(rep_map, name='canonical_index')

    # 3) build a full map (rows not in any component map to themselves)
    full_map = pd.Series(items_df.index, index=items_df.index, name='canonical_index')
    full_map.update(duplicate_to_canonical)

    # 4) keep only canonical rows; drop the rest
    keep_idx = set(full_map.unique())  # the canonical indices to keep
    items_df_deduped = items_df.loc[items_df.index.isin(keep_idx)].copy()

    # (optional) if you also want the explicit list to drop:
    to_drop = [i for i, c in full_map.items() if i != c]
    print(f"near-duplicate groups: {len(components)} | dropped rows: {len(to_drop)}")

    return items_df_deduped