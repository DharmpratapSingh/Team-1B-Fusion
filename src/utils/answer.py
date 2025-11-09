from typing import Any, Dict, List


def ensure_mtco2(rows: List[Dict[str, Any]]) -> None:
    for r in rows or []:
        if "MtCO2" not in r and isinstance(r.get("emissions_tonnes"), (int, float)):
            r["MtCO2"] = r["emissions_tonnes"] / 1e6


def deterministic_summary(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict) or "rows" not in result:
        return "No data available."
    rows: List[Dict[str, Any]] = result.get("rows", [])
    if not rows:
        return "No data found for the requested criteria."

    ensure_mtco2(rows)
    meta = result.get("meta", {})
    table_id = meta.get("table_id", "?")

    parts: List[str] = []
    # Top entry if applicable
    top = rows[0]
    name = top.get("city_name") or top.get("admin1_name") or top.get("country_name") or "the region"
    val = top.get("MtCO2")
    if isinstance(val, (int, float)):
        parts.append(f"Largest listed value appears in {name} (~{val:.1f} MtCO₂).")

    # Year range if present
    years = [r.get("year") for r in rows if isinstance(r.get("year"), int)]
    if years:
        parts.append(f"Years in this slice: {min(years)}–{max(years)}.")

    # Simple count
    parts.append(f"Listed rows: {min(len(rows), 10)} of {result.get('row_count', len(rows))}.")

    parts.append(f"Source: {table_id}, EDGAR v2024 transport. Units: tonnes CO₂ (MtCO₂).")
    return " ".join(parts)











