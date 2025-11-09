from typing import Any, Dict, List


class FallbackTrace(dict):
    @staticmethod
    def start() -> "FallbackTrace":
        return FallbackTrace(steps=[])

    def add(self, step: str, detail: Dict[str, Any] | None = None) -> None:
        self.setdefault("steps", []).append({"step": step, "detail": detail or {}})


def is_empty(result: Dict[str, Any]) -> bool:
    return isinstance(result, dict) and ("rows" in result) and (not result["rows"])


def fuzzy_where(where: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (where or {}).items():
        if isinstance(v, str) and v.strip():
            out[k] = {"contains": v.strip()}
        else:
            out[k] = v
    return out


def level_of(fid: str) -> str:
    if ("-city-" in fid) or ("_city_" in fid):
        return "city"
    if ("-admin1-" in fid) or ("_admin1_" in fid):
        return "admin1"
    return "country"


def switch_level_down(fid: str) -> str:
    if ("-city-" in fid) or ("_city_" in fid):
        fid = fid.replace("-city-", "-admin1-").replace("_city_", "_admin1_")
        return fid
    if ("-admin1-" in fid) or ("_admin1_" in fid):
        fid = fid.replace("-admin1-", "-country-").replace("_admin1_", "_country_")
        return fid
    return fid


def strip_place_filters(where: Dict[str, Any], target_level: str) -> Dict[str, Any]:
    if not isinstance(where, Dict):
        return where
    new_where = dict(where)
    if target_level == "admin1":
        new_where.pop("city_name", None)
        new_where.pop("city_id", None)
    if target_level == "country":
        new_where.pop("city_name", None)
        new_where.pop("city_id", None)
        new_where.pop("admin1_name", None)
        new_where.pop("admin1_geoid", None)
    return new_where











