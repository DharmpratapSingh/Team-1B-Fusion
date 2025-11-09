import requests
import pandas as pd

# --- Config ---
BASE = "https://api.electricitymaps.com/v3"
ZONE = "DE"   # your selected zone
START = "2023-01-01T00:00:00Z"
END   = "2023-01-03T00:00:00Z"

# --- API key (hardcoded for quick test) ---
TOKEN = "l9jJcH3gP70dvlD01GCx"
HEADERS = {"auth-token": TOKEN}

# --- Fetch carbon intensity history ---
params = {"zone": ZONE, "start": START, "end": END}
resp = requests.get(f"{BASE}/carbon-intensity/history", headers=HEADERS, params=params, timeout=60)
resp.raise_for_status()

data = resp.json().get("history", [])
df = pd.DataFrame(data)

# --- Inspect ---
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
print(df.head())