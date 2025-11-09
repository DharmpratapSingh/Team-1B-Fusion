import os, sys, json, requests, textwrap
from typing import Any, Dict, List, Union
from dotenv import load_dotenv
from pathlib import Path
from requests.auth import HTTPBasicAuth

# Load .env
load_dotenv()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://erasmus.ai/models/climategpt_8b_test/v1")
MODEL = os.getenv("MODEL", "/cache/climategpt_8b_test")
API_KEY = os.getenv("OPENAI_API_KEY", "")  # expects "username:password" (e.g., ai:4climate)
USER, PASS = API_KEY.split(":", 1) if ":" in API_KEY else ("", "")

MCP_PORT = os.getenv("PORT", "8010")
MCP_BASE = f"http://127.0.0.1:{MCP_PORT}"
SYSTEM = Path("system_prompt.txt").read_text(encoding="utf-8") if os.path.exists("system_prompt.txt") else ""

if not SYSTEM.strip():
    SYSTEM = """
    You are ClimateGPT, a data-grounded assistant that must control tools by returning JSON object(s).

    CRITICAL RULES:
    1. Return ONLY JSON - no explanations or prose
    2. Use EXACT column names from the schema below
    3. Use EXACT file_id format (see examples)
    4. If query fails or returns errors, NEVER fabricate data
    5. For comparisons/complex queries, return multiple tool calls (see MULTIPLE TOOL CALLS section)
    6. For arrays: Return on ONE LINE like [{"tool":...},{"tool":...}] NOT on separate lines

    AVAILABLE TOOLS:
      - "list_files" - List available datasets
      - "get_schema" - Get columns for a specific file_id
      - "query" - Query emissions data
      - "metrics.yoy" - Calculate year-over-year changes

    EXACT COLUMN NAMES (use these exactly):
      - Location columns: country_name, admin1_name, city_name
      - Data columns: year, emissions_tonnes
      - For monthly: add "month" column
      - DO NOT use "sector" - sector is in the file_id, not a column

    FILE_ID FORMAT:
      Pattern: {sector}-{level}-{grain}

      Examples:
        "transport-country-year"      (Germany, USA, etc.)
        "transport-admin1-year"       (California, Texas, etc.)
        "transport-city-year"         (Tokyo, London, etc.)
        "power-country-month"         (monthly power data)
        "waste-admin1-year"           (state waste data)

      Available sectors: transport, power, waste, agriculture, buildings,
                        fuel-exploitation, industrial-combustion, industrial-processes

    TOOL SCHEMAS:

    1. list_files:
       {"tool":"list_files","args":{}}

    2. get_schema:
       {"tool":"get_schema","args":{"file_id":"transport-country-year"}}

    3. query:
       {"tool":"query","args":{
         "file_id":"transport-country-year",
         "select":["country_name","year","emissions_tonnes"],
         "where":{"country_name":"Germany","year":2023},
         "group_by":[],
         "order_by":"emissions_tonnes DESC",
         "limit":10
       }}

    4. metrics.yoy:
       {"tool":"metrics.yoy","args":{
         "file_id":"transport-country-year",
         "key_col":"country_name",
         "value_col":"emissions_tonnes",
         "base_year":2019,
         "compare_year":2020,
         "top_n":10,
         "direction":"drop"
       }}

    COMMON QUERY PATTERNS:

    Simple lookup:
    {"tool":"query","args":{"file_id":"transport-country-year","select":["country_name","year","emissions_tonnes"],"where":{"country_name":"Germany","year":2023}}}

    State/admin1 query:
    {"tool":"query","args":{"file_id":"transport-admin1-year","select":["admin1_name","year","emissions_tonnes"],"where":{"admin1_name":"California","year":2022}}}

    City query:
    {"tool":"query","args":{"file_id":"transport-city-year","select":["city_name","year","emissions_tonnes"],"where":{"city_name":"Tokyo","year":2021}}}

    Monthly query:
    {"tool":"query","args":{"file_id":"power-country-month","select":["country_name","year","month","emissions_tonnes"],"where":{"country_name":"France","year":2023}}}
    
    Monthly query for state/admin1:
    {"tool":"query","args":{"file_id":"transport-admin1-month","select":["admin1_name","year","month","emissions_tonnes"],"where":{"admin1_name":"California","year":2023}}}

    RANKING QUERIES (Top/Highest/Lowest):

    Top 5 countries:
    {"tool":"query","args":{"file_id":"agriculture-country-year","select":["country_name","year","emissions_tonnes"],"where":{"year":2022},"order_by":"emissions_tonnes DESC","limit":5}}

    Highest state (DO NOT filter to specific state first!):
    {"tool":"query","args":{"file_id":"power-admin1-year","select":["admin1_name","year","emissions_tonnes"],"where":{"year":2022},"order_by":"emissions_tonnes DESC","limit":1}}

    MULTIPLE TOOL CALLS (for comparisons or multi-sector queries):

    For questions asking to "compare" or "which" between multiple entities, ALWAYS return an array:
    
    Compare USA vs China (return array - ALL ON ONE LINE):
    [{"tool":"query","args":{"file_id":"transport-country-year","select":["country_name","year","emissions_tonnes"],"where":{"country_name":"USA","year":2022}}},{"tool":"query","args":{"file_id":"transport-country-year","select":["country_name","year","emissions_tonnes"],"where":{"country_name":"China","year":2022}}}]

    Compare NYC vs LA (return array - ALL ON ONE LINE):
    [{"tool":"query","args":{"file_id":"waste-city-year","select":["city_name","year","emissions_tonnes"],"where":{"city_name":"New York City","year":2022}}},{"tool":"query","args":{"file_id":"waste-city-year","select":["city_name","year","emissions_tonnes"],"where":{"city_name":"Los Angeles","year":2022}}}]

    Multi-sector total for Germany (return array - ALL ON ONE LINE):
    [{"tool":"query","args":{"file_id":"transport-country-year","select":["country_name","year","emissions_tonnes"],"where":{"country_name":"Germany","year":2023}}},{"tool":"query","args":{"file_id":"power-country-year","select":["country_name","year","emissions_tonnes"],"where":{"country_name":"Germany","year":2023}}},{"tool":"query","args":{"file_id":"waste-country-year","select":["country_name","year","emissions_tonnes"],"where":{"country_name":"Germany","year":2023}}}]

    CRITICAL GROUP BY RULES (READ CAREFULLY):
    - ❌ DO NOT use group_by unless you are using aggregate functions (SUM, COUNT, AVG, MAX, MIN)
    - ❌ DO NOT use group_by for filtering monthly/yearly data
    - ❌ DO NOT use group_by for ranking/sorting
    - ❌ NEVER use group_by with non-aggregated columns in SELECT
    - ✅ ONLY use group_by when SELECT contains aggregates like SUM(emissions_tonnes)

    WRONG examples (DO NOT DO THIS):
    ❌ {"select":["month","emissions_tonnes"],"group_by":["month"]}
    ❌ {"select":["admin1_name","year","month","emissions_tonnes"],"where":{"admin1_name":"California"},"group_by":["month"]}
    ❌ {"select":["country_name","emissions_tonnes"],"where":{"year":2022},"group_by":["country_name"],"order_by":"emissions_tonnes DESC"}

    RIGHT examples:
    ✅ {"select":["month","emissions_tonnes"],"where":{"admin1_name":"California","year":2023}}
    ✅ {"select":["admin1_name","year","month","emissions_tonnes"],"where":{"admin1_name":"California","year":2023}}
    ✅ {"select":["country_name","year","emissions_tonnes"],"where":{"year":2022},"order_by":"emissions_tonnes DESC","limit":5}
    ✅ {"select":["country_name","SUM(emissions_tonnes) as total"],"where":{"year":2022},"group_by":["country_name"]}  (only when aggregating)

    COUNTRY NAME HANDLING:
    - System accepts common country name variations (USA, UK, China, etc.)
    - You can use "USA" or "United States of America" - both work
    - You can use "UK" or "United Kingdom" - both work
    - You can use "China", "Germany", "France" etc. - natural names work
    - The system automatically normalizes common aliases to database names
    - If unsure, use the common English name for the country

    IMPORTANT LIMITATIONS:
    - Array syntax NOT supported: DO NOT use {"country_name":["USA","China"]}
    - For comparisons, use multiple tool calls instead (see MULTIPLE TOOL CALLS above)

    REMEMBER:
    - Use hyphens in file_id: "transport-country-year" NOT "transport_country_year"
    - Use exact column names: "country_name" NOT "country"
    - Use exact column names: "admin1_name" NOT "admin1"
    - Use exact column names: "city_name" NOT "city"
    - For comparisons: return array of tool calls
    - For rankings: use order_by + limit (NO group_by)
    - DO NOT use group_by unless doing aggregation with SUM/COUNT/AVG
    """.strip()


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature
    }
    r = requests.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        auth=HTTPBasicAuth(USER, PASS) if USER else None,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def exec_single_tool(tool_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single tool call object."""
    t = tool_obj.get("tool")
    a = tool_obj.get("args", {})

    if t == "list_files":
        return requests.get(f"{MCP_BASE}/list_files").json()
    if t == "get_schema":
        fid = a["file_id"]
        return requests.get(f"{MCP_BASE}/get_schema/{fid}").json()
    if t == "query":
        # Enable server-side assist by default and normalize legacy args
        if isinstance(a, dict):
            a.setdefault("assist", True)
            fid = a.get("file_id", "")
            if "_" in fid and "-" not in fid:
                a["file_id"] = fid.replace("_", "-")

            # Normalize column names in select/where
            col_map = {
                "city": "city_name",
                "state": "admin1_name",
                "admin": "admin1_name",
                "country": "country_name",
            }
            if isinstance(a.get("select"), list):
                a["select"] = [col_map.get(c, c) for c in a["select"]]
            if isinstance(a.get("where"), dict):
                a["where"] = {col_map.get(k, k): v for k, v in a["where"].items()}

            # Normalize common country name aliases (client-side fallback)
            # MCP server's assist mode should handle this, but we add safety layer
            if isinstance(a.get("where"), dict) and "country_name" in a["where"]:
                country_aliases = {
                    "USA": "United States of America",
                    "US": "United States of America",
                    "U.S.": "United States of America",
                    "U.S.A.": "United States of America",
                    "United States": "United States of America",
                    "UK": "United Kingdom",
                    "Britain": "United Kingdom",
                    "Great Britain": "United Kingdom",
                }
                country_val = a["where"]["country_name"]
                if isinstance(country_val, str):
                    # Check if it's a known alias (case-insensitive)
                    normalized = country_aliases.get(country_val, country_aliases.get(country_val.strip()))
                    if normalized:
                        a["where"]["country_name"] = normalized
        return requests.post(f"{MCP_BASE}/query", json=a).json()
    if t in ("metrics.yoy", "yoy"):
        return requests.post(f"{MCP_BASE}/metrics/yoy", json=a).json()

    return {"error": f"unknown tool '{t}'"}

def exec_tool_call(tool_json: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Accepts a JSON string like:
    {"tool":"query","args":{...}}  - single tool call
    OR
    [{"tool":"query","args":{...}}, {"tool":"query","args":{...}}]  - multiple tool calls

    Returns single result dict or list of result dicts.
    """
    # Try to parse as JSON
    try:
        obj = json.loads(tool_json)
    except json.JSONDecodeError:
        # Initial parse failed - try various extraction methods
        
        # First try: Check if it's a properly formatted array
        trimmed = tool_json.strip()
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                obj = json.loads(trimmed)
            except:
                pass  # Fall through to other methods
        else:
            # Not a simple array, check for multiple objects
            objects = []
            
            # Method 1: Check if multiple objects on separate lines
            lines = [line.strip() for line in tool_json.split('\n') if line.strip().startswith('{')]
            if len(lines) > 1:
                for line in lines:
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "tool" in parsed:
                            objects.append(parsed)
                    except json.JSONDecodeError:
                        continue
            
            # Method 2: Check for comma-separated objects on same line
            elif tool_json.count('{"tool"') > 1:
                current_pos = 0
                while current_pos < len(tool_json):
                    if tool_json[current_pos] == '{':
                        brace_count = 0
                        start_pos = current_pos
                        found_match = False
                        for i in range(current_pos, len(tool_json)):
                            if tool_json[i] == '{':
                                brace_count += 1
                            elif tool_json[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    try:
                                        parsed = json.loads(tool_json[start_pos:i+1])
                                        if isinstance(parsed, dict) and "tool" in parsed:
                                            objects.append(parsed)
                                    except:
                                        pass
                                    current_pos = i + 1
                                    found_match = True
                                    break
                        if not found_match:
                            break
                    else:
                        current_pos += 1
            
            if objects:
                obj = objects
            else:
                # Fall back: try to extract single object
                start = tool_json.find("{")
                end = tool_json.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        obj = json.loads(tool_json[start:end+1])
                    except:
                        return {"error": "Model did not return valid JSON tool call.", "raw": tool_json}
                else:
                    return {"error": "Model did not return valid JSON tool call.", "raw": tool_json}

    # Handle array of tool calls
    if isinstance(obj, list):
        results = []
        for i, tool_obj in enumerate(obj):
            if not isinstance(tool_obj, dict) or "tool" not in tool_obj:
                results.append({"error": f"Invalid tool call at index {i}", "obj": tool_obj})
            else:
                results.append(exec_single_tool(tool_obj))
        return results

    # Handle single tool call
    elif isinstance(obj, dict) and "tool" in obj:
        return exec_single_tool(obj)

    else:
        return {"error": "Invalid tool call format. Expected dict with 'tool' key or list of such dicts.", "obj": obj}

def _ensure_mt(df_rows: Any) -> Any:
    try:
        if isinstance(df_rows, list) and df_rows and isinstance(df_rows[0], dict):
            for r in df_rows:
                if "MtCO2" not in r and "emissions_tonnes" in r and isinstance(r["emissions_tonnes"], (int, float)):
                    r["MtCO2"] = r["emissions_tonnes"] / 1e6
    except Exception:
        pass
    return df_rows

def summarize(result: Union[Dict[str, Any], List[Dict[str, Any]]], question: str) -> str:
    """Summarize query result(s) into natural language answer."""

    # Handle multiple results (from multiple tool calls)
    if isinstance(result, list):
        # Check if any results are errors
        errors = [r for r in result if isinstance(r, dict) and "error" in r]
        if errors:
            error_msgs = [r.get('detail', r.get('error', 'Query failed')) for r in errors]
            return f"Unable to retrieve some data: {'; '.join(error_msgs)}. Please check the query parameters."

        # Combine all results for summarization
        combined_data = []
        sources = set()
        for r in result:
            if isinstance(r, dict) and "rows" in r:
                r["rows"] = _ensure_mt(r["rows"])
                combined_data.extend(r["rows"])
                if "meta" in r and "table_id" in r["meta"]:
                    sources.add(r["meta"]["table_id"])

        if not combined_data:
            return "No data found for the specified queries."

        # Create combined result object
        preview_obj = {
            "rows": combined_data,
            "row_count": len(combined_data),
            "sources": list(sources)
        }
        rows_preview = json.dumps(preview_obj, ensure_ascii=False)
        source_str = f"Sources: {', '.join(sources)}, EDGAR v2024" if sources else "Source: EDGAR v2024"

    # Handle single result
    else:
        preview_obj = dict(result)
        if isinstance(preview_obj, dict) and "rows" in preview_obj:
            preview_obj["rows"] = _ensure_mt(preview_obj["rows"])
        rows_preview = json.dumps(preview_obj, ensure_ascii=False)

        # Check if query returned an error
        if isinstance(result, dict) and "error" in result:
            return f"Unable to retrieve data: {result.get('detail', result.get('error', 'Query failed'))}. Please check the query parameters or use /get_schema to verify column names."

        # Check if query returned no data
        if isinstance(result, dict) and "rows" in result and len(result["rows"]) == 0:
            return "No data found for the specified query. The location, year, or sector may not have data available in the database."

        source_str = f"Source: {result.get('meta',{}).get('table_id','?')}, EDGAR v2024"

    # Use a different system prompt for summarization (not tool calling)
    summary_system_prompt = """You are a helpful assistant that provides clear, concise answers based ONLY on the data provided.

CRITICAL RULES:
1. ONLY report facts present in the JSON data
2. NEVER add context, explanations, or background information not in the data
3. NEVER fabricate or guess values
4. If data is missing or incomplete, state that clearly
5. Keep responses concise (2-4 sentences maximum)
6. Do not return JSON or tool calls - write in natural language
7. For comparisons, clearly state values for each entity being compared"""

    prompt = textwrap.dedent(f"""
    Question: {question}

    IMPORTANT: Use ONLY the data in the JSON below. Do NOT add any context, explanations, or information not present in this data.

    Tasks:
    - State the emissions value(s) from the data
    - Convert to MtCO₂ if the value is large (divide tonnes by 1,000,000)
    - Include location, year, and sector for each value
    - For comparisons, clearly compare the values
    - Cite source: "{source_str}"
    - Be concise (2-4 sentences)
    - Do NOT add policy context, trends, or explanations not in the data

    Data (JSON):
    {rows_preview}
    """).strip()

    return chat(summary_system_prompt, prompt, temperature=0.2)

def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python run_llm.py "<your question>"')
        print('Example: uv run python run_llm.py "Which US state had the biggest drop in 2020 vs 2019?"')
        sys.exit(1)

    question = sys.argv[1]

    # 1) Ask the model to return ONLY a tool call JSON (or array of tool calls)
    tool_call = chat(
        SYSTEM,
        question + "\n\nReturn ONLY a tool call JSON (or array of tool calls for comparisons) as per the schema. Do not include any prose."
    ).strip()

    # If the model didn't return JSON, nudge once with a strict reminder
    if not ((tool_call.startswith("{") and '"tool"' in tool_call) or (tool_call.startswith("[") and '"tool"' in tool_call)):
        tool_call = chat(
            SYSTEM,
            "Return ONLY a tool call JSON (or array for comparisons) for this question:\n" + question + "\n\nReminder: never use SQL strings; use the JSON schema described."
        ).strip()

    print("\n--- TOOL CALL ---")
    print(tool_call)

    # 2) Execute tool call(s)
    result = exec_tool_call(tool_call)

    print("\n--- TOOL RESULT (first 10 rows per query) ---")
    if isinstance(result, list):
        # Multiple tool calls
        for i, res in enumerate(result):
            print(f"\n--- Result {i+1}/{len(result)} ---")
            if isinstance(res, dict) and "rows" in res:
                preview = {**res, "rows": res["rows"][:10]}
                print(json.dumps(preview, indent=2)[:2000])
            else:
                print(json.dumps(res, indent=2)[:2000])
    elif isinstance(result, dict) and "rows" in result:
        # Single tool call with rows
        preview = {**result, "rows": result["rows"][:10]}
        print(json.dumps(preview, indent=2)[:2000])
    else:
        # Single tool call with error or other result
        print(json.dumps(result, indent=2)[:2000])

    # 3) Summarize via LLM
    answer = summarize(result, question)
    print("\n=== ANSWER ===")
    print(answer)

if __name__ == "__main__":
    main()