"""Auto Trial Selector — insan müdahalesi olmadan en iyi trial'ı seçer."""


def select_best(trials: list[dict], method: str = "ratio") -> dict:
    """En iyi trial'ı otomatik seç.

    Args:
        trials: Trial sonuçları listesi (oos_net, dd, trades keys gerekli)
        method: "ratio" (net/DD) veya "score" (raw Optuna score)

    Returns:
        Seçilen trial dict'i (ratio field eklenir)
    """
    # Ratio hesapla
    for t in trials:
        t["ratio"] = round(t["oos_net"] / t["dd"], 2) if t.get("dd", 0) > 0 else 0.0

    if method == "ratio":
        by_metric = sorted(trials, key=lambda x: x["ratio"], reverse=True)
    else:
        by_metric = sorted(trials, key=lambda x: x.get("score", 0), reverse=True)

    # Filtrele: OOS pozitif, en az 10 trade, DD > 0
    valid = [t for t in by_metric if t.get("oos_net", 0) > 0 and t.get("dd", 0) > 0 and t.get("trades", 0) >= 10]

    if not valid:
        # Fallback: DD > 0 olan herhangi biri
        valid = [t for t in by_metric if t.get("dd", 0) > 0]

    if not valid:
        valid = by_metric  # absolute fallback

    return valid[0] if valid else trials[0]


def get_top_n(trials: list[dict], n: int = 10, method: str = "ratio") -> list[dict]:
    """En iyi N trial'ı döndür."""
    for t in trials:
        t["ratio"] = round(t["oos_net"] / t["dd"], 2) if t.get("dd", 0) > 0 else 0.0

    if method == "ratio":
        return sorted(trials, key=lambda x: x["ratio"], reverse=True)[:n]
    else:
        return sorted(trials, key=lambda x: x.get("score", 0), reverse=True)[:n]
