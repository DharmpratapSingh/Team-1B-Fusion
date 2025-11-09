from typing import Dict


def level_to_segment(level: str) -> str:
    if level == "city":
        return "city"
    if level == "admin1":
        return "admin1"
    return "country"


def grain_to_segment(grain: str) -> str:
    return "month" if grain == "month" else "year"


def sector_to_prefix(sector: str) -> str:
    return sector.replace(" ", "-")


def route_file_id(intent: Dict) -> str:
    sector = sector_to_prefix(intent.get("sector", "transport"))
    level = level_to_segment(intent.get("level", "country"))
    grain = grain_to_segment(intent.get("grain", "year"))
    # manifest uses hyphenated ids, e.g. transport-country-year
    return f"{sector}-{level}-{grain}"











