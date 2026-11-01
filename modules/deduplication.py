import pandas as pd
from typing import Union, List, Dict, Any

def dedupe_and_map_versions(
    df: pd.DataFrame,
    subset: Union[str, List[str]] = "tokens_nouns__unique",
    content_col: str = "Item content",
    keep: str = "first",
) -> tuple[pd.DataFrame, Dict[Any, list]]:
    """
    Deduplicate like df.drop_duplicates(subset=subset, keep=keep) and also build:
      mapping: { Item content -> [ versions_list_for_group1, versions_list_for_group2, ... ] }

    If an Item content appears in only one group, the value is a single-element list.
    """
    df = df.copy()
    subset = [subset] if isinstance(subset, str) else list(subset)

    # Make values hashable for grouping/duplicated checks
    def _hashable(v):
        if isinstance(v, list): return tuple(v)
        if isinstance(v, set):  return tuple(sorted(v))
        if isinstance(v, dict): return tuple(sorted(v.items()))
        return v

    key_df = df[subset].applymap(_hashable)
    df["_key"] = key_df.iloc[:, 0] if len(subset) == 1 else list(map(tuple, key_df.to_numpy()))

    # Representatives (same semantics as drop_duplicates(..., keep=keep))
    deduped = df.loc[~df.duplicated("_key", keep=keep)].copy()

    # All "Item content" versions for each group _key (order-preserving & unique)
    versions_by_key = (
        df.groupby("_key")[content_col]
          .apply(lambda s: list(dict.fromkeys(s.tolist())))
    )

    deduped["Item content__versions"] = deduped["_key"].map(versions_by_key)

    # ---- FIX: keep Item content as key but don't overwrite when multiple groups share it ----
    # For each Item content in the deduped rows, collect all group keys in order,
    # then map each key to its versions list.
    keys_by_content = deduped.groupby(content_col)["_key"].apply(list)
    mapping: Dict[Any, list] = {
        content: [versions_by_key[k] for k in key_list]
        for content, key_list in keys_by_content.items()
    }
    # -----------------------------------------------------------------------------------------

    deduped.drop(columns=["_key"], inplace=True)
    return deduped, mapping
