from typing import Dict, Optional, Tuple


DEFAULT_SECTOR = "transport"


def normalize_country_name(name: str) -> str:
    mapping = {
        "United States": "United States of America",
        "USA": "United States of America",
        "US": "United States of America",
        "China": "People's Republic of China",
        "Russia": "Russian Federation",
        "UK": "United Kingdom",
        "South Korea": "Republic of Korea",
        "North Korea": "Democratic People's Republic of Korea",
    }
    return mapping.get(name, name)


def parse_intent(question: str) -> Dict:
    q = (question or "").strip()
    lower = q.lower()

    # grain
    grain = "month" if "monthly" in lower or "per month" in lower else "year"

    # sector (default transport)
    sector_candidates = [
        "transport",
        "power",
        "waste",
        "agriculture",
        "buildings",
        "fuel exploitation",
        "industrial combustion",
        "industrial processes",
    ]
    sector = DEFAULT_SECTOR
    for s in sector_candidates:
        if s in lower:
            sector = s.replace(" ", "-")
            break

    # naive place extraction (heuristic): take last capitalized token sequence
    place: Optional[str] = None
    tokens = q.split()
    current: list[str] = []
    for t in tokens:
        if t[:1].isupper():
            current.append(t.strip(",.?"))
        elif current:
            place = " ".join(current)
            current = []
    if current:
        place = " ".join(current)

    # level guess: prefer city for single-word city-like, else admin1 if "state"/"province", else country
    level = "country"
    if place:
        if any(w in lower for w in ["state", "province"]):
            level = "admin1"
        elif len(place.split()) == 1 or "city" in lower:
            level = "city"

    # year extraction (single year)
    year: Optional[int] = None
    for t in tokens:
        if t.isdigit() and len(t) == 4:
            try:
                yr = int(t)
                if 1900 <= yr <= 2100:
                    year = yr
            except Exception:
                pass

    return {
        "sector": sector,
        "grain": grain,
        "place": normalize_country_name(place) if place else None,
        "level": level,
        "year": year,
    }











