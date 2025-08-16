import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
