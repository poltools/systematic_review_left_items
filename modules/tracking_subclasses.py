def plot_two_prevalence_panels(
    df1, df2,
    config1: dict, config2: dict,
    window_size: int = 5,
    normalize: bool = True,
    cumulative: bool = False,
    smooth: bool = False,
    smoothing_window: int = 2,
    figsize: tuple = (12, 8),
    style: str = "nature",
    save_path: str | None = None,
    titles: tuple[str, str] = ("Top Panel", "Bottom Panel")
):
    """
    Plot two stacked time-series panels that show, per time window, how prevalent each category is.

    What “prevalence” means here:
      - The input is a document-level dataset with a column listing categories per document.
      - We ask, for each time window (e.g., 5-year bins), how many UNIQUE documents had
        AT LEAST ONE item from each category.
      - Optionally, we normalize by the number of unique documents in the window to get
        the share of documents (i.e., prevalence proportion) rather than raw counts.

    Panels:
      - The top panel visualizes data from df1 using config1.
      - The bottom panel visualizes data from df2 using config2.

    Expected config dict keys (for each of config1/config2):
      - "year_col":    str, column with the document’s year (int-like).
      - "cat_col":     str, column with the document’s categories (iterable; will be exploded).
      - "doc_id_col":  str, unique document identifier column.
      - "rename_dict": Optional[dict], mapping from original category names to display names.
      - "category_order": Optional[list[str]], order of categories to plot (subset or superset of columns).

    Parameters
    ----------
    window_size : int
        Size of the year bin (e.g., 5 = 5-year windows: 1990–1994, 1995–1999, ...).
    normalize : bool
        If True, divide per-category document counts by the number of unique documents in the window,
        yielding a proportion in [0, 1+] (can exceed 1 if documents can belong to multiple categories).
        NOTE: If True, the `cumulative` flag is ignored by the current logic.
    cumulative : bool
        If True (and normalize is False), plot the cumulative sum of document counts across windows.
    smooth : bool
        If True, apply a rolling mean of width `smoothing_window` to the series.
    smoothing_window : int
        Window width for the rolling mean (min_periods=1).
    figsize : tuple
        Matplotlib figure size.
    style : {"nature", "default"}
        Simple styling preset for the plots.
    save_path : Optional[str]
        If provided, saves the figure to "figures/{save_path}" at 300 dpi.
    titles : tuple[str, str]
        Titles for the top and bottom panels.

    Returns
    -------
    None
        Displays the plot (and optionally saves it). No data is returned.

    Notes
    -----
    - Because documents may belong to multiple categories, normalized lines can sum to >1 in a window.
    - Each document is counted at most once per category per window (duplicates within a window are collapsed).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fixed palette (supports up to 8 series)
    base_colors = [
        '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c',
        'black', '#d62728', '#e377c2', '#808080'
    ]

    # ---- Styling presets ----
    if style == "nature":
        sns.set_theme(style='white')
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False
        })
        x_tick_rotation = 45
        line_width = 2.0
    else:
        plt.style.use('default')
        plt.rcParams.update({
            "axes.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.fancybox": False,
            "legend.edgecolor": "#cccccc",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "grid.linestyle": "--"
        })
        x_tick_rotation = 0
        line_width = 1.2

    def prepare_data(df, year_col, cat_col, doc_id_col, rename_dict=None):
        """
        Transform a document-level frame into a windowed category-prevalence table.

        Steps:
          1) explode categories so each row represents (doc, single category).
          2) coerce year to numeric and drop rows with missing year/category/doc_id.
          3) bin years into `window_size`-year windows via floor-division.
          4) drop duplicates on (doc_id, category, window) so each doc is counted
             at most once per category per window.
          5) compute:
             - docs_per_window: unique doc count per window (denominator if normalize=True).
             - docs_with_cat: unique doc count per (window, category) → wide table.
          6) optionally rename columns (categories) for display.
          7) choose the output:
             - cumulative (if True and normalize=False): cumulative sums across windows.
             - normalized (if True): divide by docs_per_window.
             - raw counts otherwise.
          8) optionally smooth via rolling mean.
        """
        # 1) One row per (doc, single category)
        df = df.explode(cat_col)

        # 2) Clean inputs
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df.dropna(subset=[year_col, cat_col, doc_id_col], inplace=True)

        # 3) Build year windows (e.g., 1990→1990, 1991→1990, ..., 1994→1990)
        df['Year Window'] = (df[year_col] // window_size) * window_size

        # 4) Ensure we count each doc once per category per window
        doc_cat_year = df.drop_duplicates(subset=[doc_id_col, cat_col, 'Year Window'])[
            [doc_id_col, cat_col, 'Year Window']
        ]

        # 5) Denominator and numerator tables
        docs_per_window = df.groupby('Year Window')[doc_id_col].nunique()
        docs_with_cat = (
            doc_cat_year
            .groupby(['Year Window', cat_col])[doc_id_col]
            .nunique()
            .unstack(fill_value=0)  # wide: one column per category
        )

        # 6) Optional cleanup of category labels
        if rename_dict:
            docs_with_cat = docs_with_cat.rename(columns=rename_dict)

        # 7) Pick the metric to return
        if cumulative and not normalize:
            prevalence = docs_with_cat.cumsum()
        elif normalize:
            # NOTE: when normalize=True, cumulative is ignored by this logic
            prevalence = docs_with_cat.div(docs_per_window, axis=0)
        else:
            prevalence = docs_with_cat.copy()

        # 8) Optional smoothing
        if smooth:
            prevalence = prevalence.rolling(window=smoothing_window, min_periods=1).mean()

        return prevalence

    # ---- Prepare both datasets ----
    prev1 = prepare_data(
        df1,
        config1["year_col"],
        config1["cat_col"],
        config1["doc_id_col"],
        rename_dict=config1.get("rename_dict")
    )
    prev2 = prepare_data(
        df2,
        config2["year_col"],
        config2["cat_col"],
        config2["doc_id_col"],
        rename_dict=config2.get("rename_dict")
    )

    # ---- Plotting ----
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, prev, config, title in zip(axes, [prev1, prev2], [config1, config2], titles):
        # Determine which categories to plot and in what order
        categories = config.get("category_order") or list(prev.columns)

        # Basic guard: palette only supports 8 series as defined above
        if len(categories) > len(base_colors):
            raise ValueError("Too many categories; max supported is 8.")

        # If a requested category is missing, add it as zeros for a consistent legend/order
        missing = [c for c in categories if c not in prev.columns]
        if missing:
            prev = prev.reindex(columns=list(prev.columns) + missing, fill_value=0)

        colors = [base_colors[i] for i in range(len(categories))]

        # Draw the panel’s lines
        prev[categories].plot(ax=ax, linewidth=line_width, color=colors)

        # Axis labeling depends on selected metric
        ylabel = (
            "Prevalence (share of docs)" if normalize else
            "Cumulative unique docs" if cumulative else
            "Unique docs"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='center')

        # Legend & ticks
        ax.legend(title=None, loc='upper left', frameon=(style == "default"))
        ax.tick_params(axis='x', rotation=x_tick_rotation)

    axes[-1].set_xlabel("Year")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"figures/{save_path}", dpi=300, bbox_inches='tight')
    plt.show()
