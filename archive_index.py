import pandas as pd
from Levenshtein import distance as levenshtein_distance


def top_n_similar(query_name: str, archive: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top-n archive rows most similar to query_name by Levenshtein.

    Mirrors the similarity formula from categorize_transactions.find_close_strings
    (1 - distance/max_length), but ranks unconditionally and returns the top n
    rather than filtering by a cutoff.
    """
    if len(archive) == 0:
        return archive.iloc[0:0].copy()

    q = query_name.lower()
    scored = archive.copy()
    scored["_distance"] = scored["Name"].apply(
        lambda x: levenshtein_distance(str(x).lower(), q)
    )
    scored["_max_len"] = scored["Name"].apply(
        lambda x: max(len(str(x)), len(query_name))
    )
    scored["Similarity"] = 1 - (scored["_distance"] / scored["_max_len"])
    top = scored.sort_values("Similarity", ascending=False).head(n)
    return top.drop(columns=["_distance", "_max_len"])
