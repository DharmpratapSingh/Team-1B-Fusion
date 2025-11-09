#!/usr/bin/env python3
"""
Enhanced ClimateGPT Interface with Personas
Implements persona-based responses and output control for curated answers
"""

import os
import json
import time
import requests
import threading
import streamlit as st
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import altair as alt
import logging
import re
from collections import OrderedDict
from functools import wraps
from test_response_metrics import ResponseEvaluator

load_dotenv()

# Streamlit page config MUST be first
st.set_page_config(
    page_title="ClimateGPT with Personas", 
    page_icon="🌍", 
    layout="wide",
    # Disable caching for better testing reliability
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONTENT_TYPE_JSON = "application/json"
DEFAULT_UNITS_MESSAGE = "All emissions data is in tonnes CO₂ (MtCO₂ for large values)"
USA_FULL_NAME = "United States of America"
DEFAULT_TIMEOUT = 30
CLIMATE_ANALYST = "Climate Analyst"
RESEARCH_SCIENTIST = "Research Scientist"
POLICY_ADVISOR = "Policy Advisor"
DATA_SPECIALIST = "Data Specialist"
SUSTAINABILITY_MANAGER = "Sustainability Manager"
LLM_MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "2"))
_LLM_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)
STUDENT = "Student"
FINANCIAL_ANALYST = "Financial Analyst"
URBAN_PLANNER = "Urban Planner"

# Safety check for constants
try:
    CONTENT_TYPE_JSON
except NameError:
    CONTENT_TYPE_JSON = "application/json"
    DEFAULT_UNITS_MESSAGE = "All emissions data is in tonnes CO₂ (MtCO₂ for large values)"
    USA_FULL_NAME = "United States of America"
    DEFAULT_TIMEOUT = 30
    CLIMATE_ANALYST = "Climate Analyst"
    RESEARCH_SCIENTIST = "Research Scientist"
    POLICY_ADVISOR = "Policy Advisor"
    DATA_SPECIALIST = "Data Specialist"
    SUSTAINABILITY_MANAGER = "Sustainability Manager"


def validate_user_input(query: str) -> Dict[str, Any]:
    """Enhanced user input validation with comprehensive security checks"""
    if not query or not query.strip():
        return {"valid": False, "error": "Query cannot be empty", "severity": "medium"}
    
    if len(query) > 1000:
        return {"valid": False, "error": "Query too long (max 1000 characters)", "severity": "medium"}
    
    # Enhanced security patterns
    SQL_INJECTION = "SQL injection"
    dangerous_patterns = [
        (r'<script.*?>.*?</script>', "XSS script injection", "high"),
        (r'javascript:', "JavaScript injection", "high"),
        (r'data:text/html', "Data URI XSS", "high"),
        (r'\.\./', "Path traversal", "high"),
        (r'UNION.*SELECT', SQL_INJECTION, "high"),
        (r'DROP.*TABLE', SQL_INJECTION, "high"),
        (r'DELETE.*FROM', SQL_INJECTION, "high"),
        (r'INSERT.*INTO', SQL_INJECTION, "high"),
        (r'UPDATE.*SET', SQL_INJECTION, "high"),
        (r'EXEC\s*\(', "Command injection", "critical"),
        (r'<iframe', "Iframe injection", "medium"),
        (r'onload\s*=', "Event handler injection", "medium"),
    ]
    
    for pattern, description, severity in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {"valid": False, "error": f"Security concern: {description}", "severity": severity}
    
    # Check for potential JSON injection
    if any(char in query for char in ['{', '}', '[', ']', '"', "'"]):
        natural_indicators = ['what', 'how', 'when', 'where', 'why', 'show', 'tell', 'find']
        if not any(indicator in query.lower() for indicator in natural_indicators):
            return {"valid": False, "error": "Query contains suspicious characters", "severity": "low"}
    
    return {"valid": True, "error": None, "severity": "low"}

def safe_json_parse(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON with comprehensive error handling"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def robust_request(url: str, method: str = "GET", max_retries: int = 3, **kwargs) -> Dict[str, Any]:
    """Make robust HTTP requests with retry logic and comprehensive error handling"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=DEFAULT_TIMEOUT, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=DEFAULT_TIMEOUT, **kwargs)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            response.raise_for_status()
            return {"success": True, "data": response.json() if response.content else {}}
            
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Request timeout for {url} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))  # Exponential backoff
                continue
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Connection error for {url} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
                continue
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}")
            return {"error": f"HTTP error {e.response.status_code}: {e.response.text}", "status_code": e.response.status_code}
        except Exception as e:
            last_error = e
            logger.warning(f"Request error for {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
                continue
    
    # All retries failed
    if last_error:
        if isinstance(last_error, requests.exceptions.Timeout):
            return {"error": "Request timed out after multiple attempts. Please try again.", "retries": max_retries}
        elif isinstance(last_error, requests.exceptions.ConnectionError):
            return {"error": "Connection failed after multiple attempts. Please check your network.", "retries": max_retries}
        else:
            return {"error": f"Request failed after {max_retries} attempts: {last_error}", "retries": max_retries}
    
    return {"error": "Unknown error occurred", "retries": max_retries}



# Configuration
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8010")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://erasmus.ai/models/climategpt_8b_test/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ai:4climate")
MODEL = os.environ.get("MODEL", "/cache/climategpt_8b_test")
USER, PASS = OPENAI_API_KEY.split(":", 1) if ":" in OPENAI_API_KEY else ("", "")

# Persona Definitions - Designed for Climate Analysts as Primary Users
PERSONAS = {
    CLIMATE_ANALYST: {
        "name": CLIMATE_ANALYST,
        "icon": "📈",
        "description": "Professional climate data analyst providing analytical insights and policy recommendations",
        "tone": "Analytical, professional, data-driven, policy-focused",
        "expertise": "Climate data analysis, policy assessment, regulatory compliance, emissions tracking",
        "response_style": "Comprehensive analytical reports with policy implications and actionable insights"
    },
    RESEARCH_SCIENTIST: {
        "name": RESEARCH_SCIENTIST,
        "icon": "🔬",
        "description": "Climate research specialist focused on scientific methodology and data validation",
        "tone": "Technical, precise, methodical, evidence-based",
        "expertise": "Climate modeling, statistical analysis, peer review, scientific research",
        "response_style": "Detailed scientific analysis with methodological rigor and statistical validation"
    },
    POLICY_ADVISOR: {
        "name": POLICY_ADVISOR,
        "icon": "📊",
        "description": "Climate policy expert providing strategic guidance and regulatory insights",
        "tone": "Strategic, professional, policy-oriented, regulatory-focused",
        "expertise": "Climate policy, regulatory frameworks, international agreements, compliance",
        "response_style": "Policy briefs with regulatory context and strategic recommendations"
    },
    DATA_SPECIALIST: {
        "name": DATA_SPECIALIST,
        "icon": "💻",
        "description": "Climate data expert specializing in data quality, processing, and visualization",
        "tone": "Technical, precise, data-focused, systematic",
        "expertise": "Data processing, quality assurance, visualization, database management",
        "response_style": "Technical data analysis with quality metrics and processing insights"
    },
    SUSTAINABILITY_MANAGER: {
        "name": SUSTAINABILITY_MANAGER,
        "icon": "🌱",
        "description": "Corporate sustainability professional focused on implementation and reporting",
        "tone": "Practical, implementation-focused, results-oriented, corporate",
        "expertise": "Corporate sustainability, ESG reporting, implementation strategies, stakeholder engagement",
        "response_style": "Implementation-focused analysis with corporate sustainability insights"
    },
    STUDENT: {
        "name": STUDENT,
        "icon": "🎓",
        "description": "Learner seeking clear, simple explanations of climate emissions data",
        "tone": "Clear, friendly, accessible, analogy-driven",
        "expertise": "Basic understanding of trends, rankings, and comparisons",
        "response_style": "Step-by-step explanations using plain language and concrete examples"
    },
    FINANCIAL_ANALYST: {
        "name": FINANCIAL_ANALYST,
        "icon": "💼",
        "description": "Analyst focused on risk/opportunity signals from emissions trends and concentration",
        "tone": "Concise, risk-aware, signal-focused",
        "expertise": "Trend direction, momentum hints, concentration (top-N), policy exposure proxies",
        "response_style": "Bullet-like takeaways on trends/rankings with cautious qualifiers"
    },
    URBAN_PLANNER: {
        "name": URBAN_PLANNER,
        "icon": "🏙️",
        "description": "Planner focused on city/state patterns, seasonality, and hotspots",
        "tone": "Practical, action-oriented, local-scale",
        "expertise": "City/admin1 monthly seasonality, peaks/valleys, intra-region hotspots",
        "response_style": "Actionable notes highlighting monthly peaks and local hotspots"
    },
    "No Persona": {
        "name": "No Persona",
        "icon": "🤖",
        "description": "Basic AI assistant providing straightforward, unfiltered responses",
        "tone": "Neutral, direct, factual",
        "expertise": "General data analysis and reporting",
        "response_style": "Straightforward data presentation without specialized framing"
    }
}

# Persona-specific system prompts
def get_persona_system_prompt(persona_key: str) -> str:
    """Get persona-specific system prompt"""
    base_prompt = """
You are ClimateGPT-Dev with a specific professional persona. You MUST answer by calling HTTP tools instead of guessing.
Data sources: EDGAR v2024 comprehensive emissions data including transport, power industry, waste, agriculture, buildings, fuel exploitation, industrial combustion, and industrial processes. Units: tonnes CO₂ (use MtCO₂ for large numbers).

CRITICAL: Choose the correct dataset based on the user's question:
- If user asks for "monthly" data → use -month datasets
- If user asks for "annual" or "yearly" data → use -year datasets  
- If user asks for "from 2020 to 2022" without specifying monthly → use -month datasets for detailed trends
- If user asks for "in 2020" → use -year datasets

Available datasets:
- Transport: transport-country-year, transport-admin1-year, transport-city-year, transport-country-month, transport-admin1-month, transport-city-month
- Power Industry: power-country-year, power-admin1-year, power-city-year, power-country-month, power-admin1-month, power-city-month
- Waste: waste-country-year, waste-admin1-year, waste-city-year, waste-country-month, waste-admin1-month, waste-city-month
- Agriculture: agriculture-country-year, agriculture-admin1-year, agriculture-city-year, agriculture-country-month, agriculture-admin1-month, agriculture-city-month
- Buildings: buildings-country-year, buildings-admin1-year, buildings-city-year, buildings-country-month, buildings-admin1-month, buildings-city-month
- Fuel Exploitation: fuel-exploitation-country-year, fuel-exploitation-admin1-year, fuel-exploitation-city-year, fuel-exploitation-country-month, fuel-exploitation-admin1-month, fuel-exploitation-city-month
- Industrial Combustion: industrial-combustion-country-year, industrial-combustion-admin1-year, industrial-combustion-city-year, industrial-combustion-country-month, industrial-combustion-admin1-month, industrial-combustion-city-month
- Industrial Processes: industrial-processes-country-year, industrial-processes-admin1-year, industrial-processes-city-year, industrial-processes-country-month, industrial-processes-admin1-month, industrial-processes-city-month

CRITICAL: Return ONLY ONE valid JSON object. Do NOT return multiple JSON objects concatenated together.

Tool call format (SINGLE JSON ONLY):
{"tool":"query","args":{"file_id":"dataset-name","select":["col1","col2"],"where":{"col":"value"},"limit":10}}

Valid file_ids:
- transport-country-year, transport-admin1-year, transport-city-year, transport-country-month, transport-admin1-month, transport-city-month
- power-country-year, power-admin1-year, power-city-year, power-country-month, power-admin1-month, power-city-month
- waste-country-year, waste-admin1-year, waste-city-year, waste-country-month, waste-admin1-month, waste-city-month
- agriculture-country-year, agriculture-admin1-year, agriculture-city-year, agriculture-country-month, agriculture-admin1-month, agriculture-city-month
- buildings-country-year, buildings-admin1-year, buildings-city-year, buildings-country-month, buildings-admin1-month, buildings-city-month
- fuel-exploitation-country-year, fuel-exploitation-admin1-year, fuel-exploitation-city-year, fuel-exploitation-country-month, fuel-exploitation-admin1-month, fuel-exploitation-city-month
- industrial-combustion-country-year, industrial-combustion-admin1-year, industrial-combustion-city-year, industrial-combustion-country-month, industrial-combustion-admin1-month, industrial-combustion-city-month
- industrial-processes-country-year, industrial-processes-admin1-year, industrial-processes-city-year, industrial-processes-country-month, industrial-processes-admin1-month, industrial-processes-city-month

Valid columns:
- country_name, iso3, year, emissions_tonnes, MtCO2
- admin1_name, admin1_geoid (for admin1 datasets)
- city_name, city_id (for city datasets)
- month (for monthly datasets)

IMPORTANT: Use exact country names from the data:
- USA_FULL_NAME (NOT "United States" or "USA")
- "People's Republic of China" (NOT "China")
- "Russian Federation" (NOT "Russia")

IMPORTANT RULES:
1. For "top N countries" queries: ALWAYS use order_by="MtCO2 DESC" and limit=N
2. For "by year" queries: include year in where clause
3. For "excluding invalid" queries: add where clause to filter out invalid data
4. ALWAYS include the requested number in limit (e.g., limit:25 for top 25)
5. Return ONLY ONE JSON object, never multiple objects

CRITICAL: Return ONLY ONE JSON object with "tool" and "args" keys. No multiple JSON objects, no markdown code blocks, no explanations, no additional text, no trailing characters, no quotes around the JSON.
"""
    
    persona = PERSONAS[persona_key]
    
    persona_specific = f"""

PERSONA: {persona['name']} {persona['icon']}
DESCRIPTION: {persona['description']}
TONE: {persona['tone']}
EXPERTISE: {persona['expertise']}
RESPONSE STYLE: {persona['response_style']}

When providing responses, embody this persona's characteristics:
- Use the specified tone and expertise areas
- Focus on the response style appropriate for this persona
- Provide insights relevant to this persona's domain
- Maintain professional credibility while being authentic to the persona
"""
    
    # Common guardrails applicable to all personas
    guardrails = """

STRICT DATA GUARDRAILS:
- Use only available fields: country_name, admin1_name, city_name, year, month, emissions_tonnes, MtCO2
- Do NOT produce metrics not present: per-capita, currency ($), costs, prices, revenues, ESG scores, GDP, forecasts.
- If asked for unsupported metrics, clearly state unavailability and offer trends/rankings/seasonality using available fields instead.

ALLOWED ANALYSES ONLY:
- Rankings (top-N), year-over-year deltas if present, multi-region comparisons, monthly seasonality where applicable.
- Prefer monthly for seasonality; prefer city/admin1 when user asks local questions.
"""

    # Persona-specific preferences
    persona_prefs = ""
    if persona_key == STUDENT:
        persona_prefs = """

STUDENT PREFERENCES:
- Explain simply with plain language; avoid heavy statistics.
- Define terms briefly; focus on big movers and simple trends.
- Prefer yearly data unless user asks about months/seasonality explicitly.
"""
    elif persona_key == FINANCIAL_ANALYST:
        persona_prefs = """

FINANCIAL ANALYST PREFERENCES:
- Focus on direction of change, concentration (top emitters), and momentum hints using YoY when available.
- Do not infer monetary impact or per-capita; stick to emissions signals and geography/sector context.
- Prefer yearly country/admin1 rankings unless user requests monthly detail.
"""
    elif persona_key == URBAN_PLANNER:
        persona_prefs = """

URBAN PLANNER PREFERENCES:
- Prefer city/admin1 monthly datasets to identify peaks/valleys and hotspots.
- Highlight local patterns and practical planning windows during high/low months.
- Roll up to country only if city/admin1 data is not requested or unavailable.
"""

    return base_prompt + persona_specific + guardrails + persona_prefs

# Output control system
class OutputController:
    """Controls and curates ClimateGPT output based on persona and user preferences"""
    
    def __init__(self):
        self.response_templates = {
            CLIMATE_ANALYST: {
                "intro": "Based on the EDGAR v2024 data analysis:",
                "data_focus": "The analytical findings indicate:",
                "conclusion": "From an analytical perspective, this data suggests:",
                "units": DEFAULT_UNITS_MESSAGE
            },
            RESEARCH_SCIENTIST: {
                "intro": "Based on the EDGAR v2024 data analysis:",
                "data_focus": "The statistical analysis reveals:",
                "conclusion": "From a scientific perspective, these findings indicate:",
                "units": DEFAULT_UNITS_MESSAGE
            },
            POLICY_ADVISOR: {
                "intro": "From a policy analysis perspective:",
                "data_focus": "The regulatory implications of this data are:",
                "conclusion": "This data suggests the following policy considerations:",
                "units": "Emissions data in tonnes CO₂ (MtCO₂ for large values)"
            },
            DATA_SPECIALIST: {
                "intro": "Based on the data processing and analysis:",
                "data_focus": "The data quality assessment shows:",
                "conclusion": "From a data management perspective, this indicates:",
                "units": DEFAULT_UNITS_MESSAGE
            },
            SUSTAINABILITY_MANAGER: {
                "intro": "From a sustainability management perspective:",
                "data_focus": "The corporate sustainability implications are:",
                "conclusion": "For sustainability strategy, this suggests:",
                "units": "Emissions data in tonnes CO₂ (MtCO₂ for large values)"
            },
            STUDENT: {
                "intro": "Let's break this down simply:",
                "data_focus": "In plain terms, the data shows:",
                "conclusion": "In summary, the simplest takeaway is:",
                "units": DEFAULT_UNITS_MESSAGE
            },
            FINANCIAL_ANALYST: {
                "intro": "From a signals perspective:",
                "data_focus": "Key emissions signals:",
                "conclusion": "Risk/opportunity takeaways (emissions only):",
                "units": DEFAULT_UNITS_MESSAGE
            },
            URBAN_PLANNER: {
                "intro": "From a local planning perspective:",
                "data_focus": "City/state patterns and seasonality:",
                "conclusion": "Planning notes based on the data:",
                "units": DEFAULT_UNITS_MESSAGE
            },
            "No Persona": {
                "intro": "Based on the EDGAR v2024 data analysis:",
                "data_focus": "The data shows:",
                "conclusion": "Summary of findings:",
                "units": DEFAULT_UNITS_MESSAGE
            }
        }
    
    def curate_response(self, raw_response: str, persona_key: str, data_context: Dict) -> str:
        """Curate and enhance the response based on persona"""
        templates = self.response_templates[persona_key]
        
        # Start with empty response (no header)
        curated_response = ""
        
        # Add the original response with persona-specific enhancements
        curated_response += raw_response
        
        # Add persona-specific conclusion if data is available
        if data_context.get('rows'):
            curated_response += f"\n\n{templates['conclusion']}\n"
            
            # Add specific insights based on persona
            if persona_key == CLIMATE_ANALYST:
                curated_response += self._add_analyst_insights(data_context)
            elif persona_key == RESEARCH_SCIENTIST:
                curated_response += self._add_scientific_insights(data_context)
            elif persona_key == POLICY_ADVISOR:
                curated_response += self._add_policy_insights(data_context)
            elif persona_key == DATA_SPECIALIST:
                curated_response += self._add_data_insights(data_context)
            elif persona_key == SUSTAINABILITY_MANAGER:
                curated_response += self._add_sustainability_insights(data_context)
            elif persona_key == STUDENT:
                curated_response += self._add_student_insights(data_context)
            elif persona_key == FINANCIAL_ANALYST:
                curated_response += self._add_financial_insights(data_context)
            elif persona_key == URBAN_PLANNER:
                curated_response += self._add_urban_planner_insights(data_context)
        
        # Add units clarification
        curated_response += f"\n\n*{templates['units']}*"
        
        return curated_response
    
    def _add_analyst_insights(self, data_context: Dict) -> str:
        """Add analytical insights for Climate Analyst persona"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        
        insights = []
        if len(rows) > 1:
            insights.append("The data shows significant variation across regions, indicating diverse emission patterns that warrant further analysis.")
        
        # Check for trends
        if 'year' in str(rows[0]):
            insights.append("Temporal analysis reveals emission trends that require monitoring and assessment.")
        
        # Check for high emitters
        if any(row.get('MtCO2', 0) > 100 for row in rows[:5]):
            insights.append("High-emitting regions present opportunities for targeted policy interventions and emissions reduction strategies.")
        
        return " ".join(insights)
    
    def _add_scientific_insights(self, data_context: Dict) -> str:
        """Add scientific insights to the response"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        
        insights = []
        if len(rows) > 1:
            insights.append("The data shows significant variation across regions, indicating diverse emission patterns.")
        
        # Check for trends
        if 'year' in str(rows[0]):
            insights.append("Temporal analysis reveals emission trends that require further investigation.")
        
        return " ".join(insights)
    
    def _add_policy_insights(self, data_context: Dict) -> str:
        """Add policy insights to the response"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        
        insights = []
        insights.append("These findings have important implications for climate policy development.")
        
        # Check for high emitters
        if any(row.get('MtCO2', 0) > 100 for row in rows[:5]):
            insights.append("High-emitting regions may require targeted policy interventions.")
        
        return " ".join(insights)
    
    def _add_data_insights(self, data_context: Dict) -> str:
        """Add data management insights for Data Specialist persona"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        
        insights = []
        insights.append("The dataset quality appears consistent with EDGAR v2024 standards.")
        
        # Check for data completeness
        if len(rows) > 3:
            insights.append("Multiple data points provide sufficient sample size for reliable analysis.")
        
        return " ".join(insights)
    
    def _add_sustainability_insights(self, data_context: Dict) -> str:
        """Add sustainability management insights"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        
        insights = []
        insights.append("These trends present both challenges and opportunities for corporate sustainability strategy.")
        
        # Check for business-relevant patterns
        if len(rows) > 3:
            insights.append("The competitive landscape shows varying emission profiles across regions.")
        
        return " ".join(insights)

    def _add_student_insights(self, data_context: Dict) -> str:
        """Add simple, explanatory insights for Student persona"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        insights = []
        # Define terms once
        insights.append("Emissions here means CO₂ released from the sector in that place and time.")
        # Biggest value among first few rows
        try:
            top = max(rows[:5], key=lambda r: r.get('MtCO2', r.get('emissions_tonnes', 0) / 1e6 if isinstance(r.get('emissions_tonnes'), (int, float)) else 0))
            name = top.get('city_name') or top.get('admin1_name') or top.get('country_name') or "the region"
            val = top.get('MtCO2')
            if not isinstance(val, (int, float)) and isinstance(top.get('emissions_tonnes'), (int, float)):
                val = top['emissions_tonnes'] / 1e6
            if isinstance(val, (int, float)):
                insights.append(f"The highest value in this slice appears in {name} (~{val:.1f} MtCO₂).")
        except Exception:
            pass
        return " ".join(insights)

    def _add_financial_insights(self, data_context: Dict) -> str:
        """Add emissions-only signals for Financial Analyst persona"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        insights = []
        # Concentration hint: top few account for majority in the slice
        try:
            vals = []
            for r in rows[:10]:
                v = r.get('MtCO2')
                if not isinstance(v, (int, float)) and isinstance(r.get('emissions_tonnes'), (int, float)):
                    v = r['emissions_tonnes'] / 1e6
                if isinstance(v, (int, float)):
                    vals.append(v)
            if vals:
                total = sum(vals)
                top3 = sum(sorted(vals, reverse=True)[:3])
                share = (top3 / total * 100) if total else 0
                insights.append(f"Concentration: top 3 in this view ≈ {share:.0f}% of listed emissions.")
        except Exception:
            pass
        # Momentum hint if multiple years present
        try:
            years = {r.get('year') for r in rows if isinstance(r.get('year'), int)}
            if len(years) >= 2:
                insights.append("Momentum: multi-year differences present; consider direction and stability (emissions only).")
        except Exception:
            pass
        return " ".join(insights)

    def _add_urban_planner_insights(self, data_context: Dict) -> str:
        """Add monthly seasonality and hotspot hints for Urban Planner persona"""
        rows = data_context.get('rows', [])
        if not rows:
            return ""
        insights = []
        # Seasonality if month exists
        if any('month' in r for r in rows if isinstance(r, dict)):
            insights.append("Seasonality: monthly patterns detected; plan for resources near peak months.")
        # Hotspot: identify entity with highest MtCO2 among slice
        try:
            top = max(rows[:10], key=lambda r: r.get('MtCO2', r.get('emissions_tonnes', 0) / 1e6 if isinstance(r.get('emissions_tonnes'), (int, float)) else 0))
            name = top.get('city_name') or top.get('admin1_name') or top.get('country_name') or "the region"
            insights.append(f"Hotspot focus: {name} shows the highest emissions within this selection.")
        except Exception:
            pass
        return " ".join(insights)

# Initialize output controller
output_controller = OutputController()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = CLIMATE_ANALYST
if "last_result" not in st.session_state:
    st.session_state.last_result = None

def chat_with_climategpt(system: str, user_message: str, temperature: float = 0.2) -> str:
    """Send message to ClimateGPT LLM and get response"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature
    }
    
    try:
        with _LLM_SEMAPHORE:
            r = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={"Content-Type": CONTENT_TYPE_JSON},
                data=json.dumps(payload),
                auth=HTTPBasicAuth(USER, PASS) if USER else None,
                timeout=120,
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error communicating with ClimateGPT: {str(e)}"

def normalize_country_name(country_name: str) -> str:
    """Normalize country names to match the data format"""
    country_mapping = {
        "United States": USA_FULL_NAME,
        "USA": USA_FULL_NAME,
        "US": USA_FULL_NAME,
        "China": "People's Republic of China",
        "Russia": "Russian Federation",
        "UK": "United Kingdom",
        "South Korea": "Republic of Korea",
        "North Korea": "Democratic People's Republic of Korea"
    }
    return country_mapping.get(country_name, country_name)

# --- Heuristic geo-entity detection (admin1/city) ---
US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming"
}

# Common city name normalizations (can be extended as needed)
COMMON_CITIES = {
    "new york city": {"city_name": "New York", "country_name": USA_FULL_NAME},
    "new york": {"city_name": "New York", "country_name": USA_FULL_NAME},
    "los angeles": {"city_name": "Los Angeles", "country_name": USA_FULL_NAME},
    "san francisco": {"city_name": "San Francisco", "country_name": USA_FULL_NAME},
    "chicago": {"city_name": "Chicago", "country_name": USA_FULL_NAME},
    "london": {"city_name": "London", "country_name": "United Kingdom"},
    "paris": {"city_name": "Paris", "country_name": "France"},
    "tokyo": {"city_name": "Tokyo", "country_name": "Japan"}
}

def detect_geo_entity(question: str) -> Dict[str, Any]:
    """Detect admin1 or city from the free-text question.
    Returns a dict like {"level": "admin1"|"city", "where": {...}} or {} if none.
    """
    q = (question or "").lower()

    # City detection (exact phrase match first)
    for key, meta in COMMON_CITIES.items():
        if key in q:
            where = {"city_name": meta["city_name"]}
            if meta.get("country_name"):
                where["country_name"] = meta["country_name"]
            return {"level": "city", "where": where}

    # Admin1 detection for US states (simple heuristic)
    for state in US_STATES:
        # Match whole-word state names
        if re.search(rf"\b{re.escape(state)}\b", q):
            # Title-case state for matching stored data
            state_tc = " ".join([w.capitalize() for w in state.split()])
            return {"level": "admin1", "where": {"admin1_name": state_tc, "country_name": USA_FULL_NAME}}

    return {}

def apply_level_constraints_to_tool(tool_json: str, constraints: Dict[str, Any]) -> str:
    """Force level-specific file_id and required where filters onto a single-tool JSON string.
    Best-effort parsing; returns updated JSON string (or original if parsing fails).
    """
    try:
        obj = json.loads(tool_json)
        if not isinstance(obj, dict):
            return tool_json
        args = obj.get("args") or obj.get("tool_args") or {}
        file_id: str = args.get("file_id", "")

        level = constraints.get("level")
        where_req = constraints.get("where", {})
        if level:
            # Preserve sector and grain from existing file_id if present; only swap the level segment
            if file_id:
                file_id = file_id.replace("-country-", f"-{level}-").replace("_country_", f"_{level}_")
                file_id = file_id.replace("-admin1-", f"-{level}-").replace("_admin1_", f"_{level}_")
                file_id = file_id.replace("-city-", f"-{level}-").replace("_city_", f"_{level}_")
                # If no level segment at all, default to yearly grain and transport sector
                if ("-country-" not in file_id and "-admin1-" not in file_id and "-city-" not in file_id):
                    # Try to infer sector prefix if present; else default transport
                    sector_prefix = (file_id.split("-")[0] or "transport") if "-" in file_id else "transport"
                    grain_suffix = "month" if "-month" in file_id or file_id.endswith("month") else "year"
                    file_id = f"{sector_prefix}-{level}-{grain_suffix}"
            else:
                # No file_id provided: choose safe default
                file_id = f"transport-{level}-year"

            args["file_id"] = file_id

        if where_req:
            where = args.get("where", {})
            if not isinstance(where, dict):
                where = {}
            # Merge required keys without overwriting explicit user filters on same key
            for k, v in where_req.items():
                where.setdefault(k, v)
            args["where"] = where

        # Ensure city/admin1 select fields exist for proper display when level constrained
        sel = args.get("select", [])
        if isinstance(sel, list):
            if constraints.get("level") == "city":
                for col in ["city_name", "country_name", "year", "MtCO2"]:
                    if col not in sel:
                        sel.append(col)
            elif constraints.get("level") == "admin1":
                for col in ["admin1_name", "country_name", "year", "MtCO2"]:
                    if col not in sel:
                        sel.append(col)
            args["select"] = sel

        if "args" in obj:
            obj["args"] = args
        else:
            obj["tool_args"] = args

        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return tool_json

# Circuit breaker for MCP server calls
class CircuitBreaker:
    def __init__(self, max_failures: int = 5, timeout: int = 60):
        self.max_failures = max_failures
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        current_time = time.time()
        
        # Reset circuit breaker if timeout has passed
        if current_time - self.last_failure_time > self.timeout and self.state == "OPEN":
            self.state = "HALF_OPEN"
            self.failures = 0
        
        # Check if circuit is open
        if self.state == "OPEN":
            return {"error": "Service temporarily unavailable (circuit breaker open)", "retry_after": self.timeout - (current_time - self.last_failure_time)}
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = current_time
            
            if self.failures >= self.max_failures:
                self.state = "OPEN"
            
            raise e

# Global circuit breaker instance
mcp_circuit_breaker = CircuitBreaker()

class SimpleLRUCache:
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._store = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None
    
    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)

# Cache for tool call responses to avoid repeated MCP hits on retries/batch
_TOOL_CACHE = SimpleLRUCache(maxsize=int(os.environ.get("MCP_TOOL_CACHE_SIZE", "256")))

def exec_tool_call(tool_json: str) -> dict:
    """Execute tool call against MCP server with caching"""
    # Clean the JSON string first
    cleaned_json = tool_json.strip()
    
    # Remove markdown code blocks if present
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[7:]
    if cleaned_json.startswith("```"):
        cleaned_json = cleaned_json[3:]
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-3]
    cleaned_json = cleaned_json.strip()
    
    # Remove any leading/trailing quotes if the entire response is quoted
    if cleaned_json.startswith('"') and cleaned_json.endswith('"'):
        cleaned_json = cleaned_json[1:-1]
        # Unescape any escaped quotes
        cleaned_json = cleaned_json.replace('\\"', '"')
    
    try:
        obj = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        # Try to extract the FIRST complete JSON object only
        start = cleaned_json.find("{")
        if start != -1:
            # Find the first complete JSON object by counting braces
            brace_count = 0
            json_end = -1
            for i, char in enumerate(cleaned_json[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                json_candidate = cleaned_json[start:json_end+1]
                try:
                    obj = json.loads(json_candidate)
                except json.JSONDecodeError as e2:
                    return {
                        "error": f"Invalid JSON format: {str(e2)}", 
                        "raw": tool_json[:200] + "..." if len(tool_json) > 200 else tool_json,
                        "debug": f"Tried to parse: {json_candidate[:200]}..."
                    }
            else:
                return {
                    "error": f"Invalid JSON format: {str(e)}", 
                    "raw": tool_json[:200] + "..." if len(tool_json) > 200 else tool_json,
                    "debug": "No complete JSON object found"
                }
        else:
            return {
                "error": f"JSON decode error: {str(e)}", 
                "raw": tool_json[:200] + "..." if len(tool_json) > 200 else tool_json,
                "debug": "No valid JSON brackets found"
            }
    
    # Handle both "tool"/"args" and "tool_call"/"tool_args" formats
    tool = obj.get("tool") or obj.get("tool_call")
    args = obj.get("args", {}) or obj.get("tool_args", {})
    
    # Define HTTP helpers with retry/backoff using robust_request
    def _http_get(url: str) -> Any:
        rr = robust_request(url, method="GET")
        if rr.get("success"):
            return rr.get("data", {})
        raise requests.exceptions.RequestException(rr.get("error", "GET failed"))
    
    def _http_post(url: str, payload: dict) -> Any:
        rr = robust_request(url, method="POST", json=payload)
        if rr.get("success"):
            return rr.get("data", {})
        raise requests.exceptions.RequestException(rr.get("error", "POST failed"))
    
    # Define MCP server operations with circuit breaker protection
    def _list_files():
        return _http_get(f"{MCP_URL}/list_files")
    
    def _get_schema(file_id):
        return _http_get(f"{MCP_URL}/get_schema/{file_id}")
    
    def _execute_query(query_args):
        # Normalize country names in the query
        if "where" in query_args and "country_name" in query_args["where"]:
            query_args["where"]["country_name"] = normalize_country_name(query_args["where"]["country_name"])
        # Enable server-side assist by default
        query_args.setdefault("assist", True)
        return _http_post(f"{MCP_URL}/query", query_args)
    
    def _execute_yoy_metrics(metrics_args):
        return _http_post(f"{MCP_URL}/metrics/yoy", metrics_args)
    
    def _execute_batch_query(batch_args):
        return _http_post(f"{MCP_URL}/batch/query", batch_args)
    
    # Execute tool with circuit breaker protection
    try:
        # Cache key based on normalized tool+args
        cache_key = None
        try:
            cache_key = json.dumps({"tool": tool, "args": args}, sort_keys=True)
        except Exception:
            cache_key = None

        if cache_key:
            cached = _TOOL_CACHE.get(cache_key)
            if cached is not None:
                return cached

        if tool == "list_files":
            result = mcp_circuit_breaker.call(_list_files)
        elif tool == "get_schema":
            if "file_id" not in args:
                return {"error": "Missing required parameter: file_id"}
            result = mcp_circuit_breaker.call(_get_schema, args["file_id"])
        elif tool == "query":
            # Primary attempt
            result = mcp_circuit_breaker.call(_execute_query, args)

            # Robust fallbacks when result is empty or error
            def _is_empty(res: dict) -> bool:
                return isinstance(res, dict) and ("rows" in res) and (not res["rows"])

            def _fuzzy_where(original_where: dict) -> dict:
                new_where = {}
                for k, v in (original_where or {}).items():
                    if isinstance(v, str) and v.strip():
                        new_where[k] = {"contains": v.strip()}
                    else:
                        new_where[k] = v
                return new_where

            def _switch_level(file_id: str, direction: str) -> str:
                # direction: "down" city->admin1->country
                if direction == "down":
                    file_id = file_id.replace("-city-", "-admin1-")
                    file_id = file_id.replace("_city_", "_admin1_")
                    if "-city-" not in file_id and "_city_" not in file_id and (
                        "-admin1-" in file_id or "_admin1_" in file_id
                    ):
                        return file_id
                    # If already admin1 or switch failed, go to country
                    file_id = file_id.replace("-admin1-", "-country-")
                    file_id = file_id.replace("_admin1_", "_country_")
                    return file_id
                return file_id

            def _strip_place_filters(where: dict, target_level: str) -> dict:
                if not isinstance(where, dict):
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

            # 1) If empty, try fuzzy contains on string filters
            if ("error" in result) or _is_empty(result):
                fuzzy_args = dict(args)
                fuzzy_args["where"] = _fuzzy_where(args.get("where", {}))
                try:
                    result2 = mcp_circuit_breaker.call(_execute_query, fuzzy_args)
                    if not ("error" in result2 or _is_empty(result2)):
                        result = result2
                except Exception:
                    pass

            # 2) If still empty and file_id is city/admin1, roll up a level
            def _level_of(fid: str) -> str:
                if ("-city-" in fid) or ("_city_" in fid):
                    return "city"
                if ("-admin1-" in fid) or ("_admin1_" in fid):
                    return "admin1"
                return "country"

            if ("error" in result) or _is_empty(result):
                current_level = _level_of(args.get("file_id", ""))
                if current_level in ("city", "admin1"):
                    rolled_args = dict(args)
                    rolled_args["file_id"] = _switch_level(args["file_id"], "down")
                    target_level = _level_of(rolled_args["file_id"])  # after switch
                    rolled_args["where"] = _strip_place_filters(args.get("where", {}), target_level)
                    try:
                        result3 = mcp_circuit_breaker.call(_execute_query, rolled_args)
                        if not ("error" in result3 or _is_empty(result3)):
                            result = result3
                    except Exception:
                        pass

            # 3) If still empty, drop filters and return a tiny default slice
            if ("error" in result) or _is_empty(result):
                loose_args = dict(args)
                loose_args.pop("where", None)
                loose_args.setdefault("limit", 5)
                try:
                    result4 = mcp_circuit_breaker.call(_execute_query, loose_args)
                    if not ("error" in result4 or _is_empty(result4)):
                        result = result4
                except Exception:
                    pass
        elif tool in ("metrics.yoy", "yoy"):
            result = mcp_circuit_breaker.call(_execute_yoy_metrics, args)
        elif tool == "batch_query":
            result = mcp_circuit_breaker.call(_execute_batch_query, args)
        else:
            return {"error": f"Unknown tool '{tool}'. Supported tools: list_files, get_schema, query, metrics.yoy, batch_query"}
        
        if cache_key and isinstance(result, dict) and "error" not in result:
            _TOOL_CACHE.set(cache_key, result)
        return result
    
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server is taking too long to respond.", "retry": True}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed. Please check if the MCP server is running.", "retry": True}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error {e.response.status_code}: {e.response.text}", "status_code": e.response.status_code}
    except Exception as e:
        logger.error(f"Unexpected error in exec_tool_call: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def create_fallback_response(question: str, error_context: Dict[str, Any]) -> str:
    """Create helpful fallback responses when errors occur"""
    error_type = error_context.get("error_type", "unknown")
    
    fallback_responses = {
        "network_error": "I'm having trouble connecting to the data server. Please try again in a moment.",
        "timeout": "The request is taking longer than expected. Please try a simpler question.",
        "data_not_found": "I couldn't find data for your specific query. This could be due to:\n- The country name not matching exactly (try 'United States of America' instead of 'USA')\n- The year not being available in the dataset\n- The sector not having data for that region/timeframe\n\nTry asking about a different country, year, or sector, or use the Debug section to test data availability.",
        "invalid_query": "I didn't understand your question. Please try rephrasing it in simpler terms.",
        "server_error": "There's a temporary issue with the data server. Please try again later.",
        "json_error": "I received an unexpected response format. Please try rephrasing your question.",
        "circuit_breaker": "The data service is temporarily unavailable. Please try again in a few minutes.",
        "llm_error": "I'm having trouble processing your question. Please try rephrasing it.",
        "empty_response": "I couldn't generate a response. Please try a different question.",
        "unexpected_error": "An unexpected error occurred. Please try again."
    }
    
    base_response = fallback_responses.get(error_type, "I encountered an unexpected error. Please try again.")
    
    # Add helpful suggestions based on the question
    suggestions = []
    if "emissions" in question.lower():
        suggestions.append("Try asking about specific countries, years, or sectors like 'transport' or 'power industry'")
    if "top" in question.lower():
        suggestions.append("Try asking for 'top 10 countries' or 'top 5 cities'")
    if "monthly" in question.lower():
        suggestions.append("Try asking for 'annual data' instead, or specify a particular year")
    if "waste" in question.lower():
        suggestions.append("Try asking about 'waste emissions' for specific countries or years")
    
    if suggestions:
        base_response += f"\n\n💡 **Suggestions:** {suggestions[0]}"
    
    return base_response

def process_climategpt_question(question: str, persona_key: str) -> Tuple[str, Dict[str, Any], str]:
    """Process question through ClimateGPT LLM workflow with persona"""
    try:
        # Get persona-specific system prompt
        system_prompt = get_persona_system_prompt(persona_key)

        # Heuristic geo detection to bias level and filters
        geo_constraints = detect_geo_entity(question)

        # If detected, add HARD CONSTRAINTS to the system prompt
        if geo_constraints:
            must_level = geo_constraints.get("level")
            must_where = {k: v for k, v in geo_constraints.get("where", {}).items()}
            where_str = ", ".join([f"\"{k}\": \"{v}\"" for k, v in must_where.items()])
            system_prompt += f"\n\nHARD CONSTRAINTS:\n- You MUST use a file_id with '-{must_level}-' (not country).\n- You MUST include where: {{{where_str}}}.\n- Do NOT change these constraints.\n"

        # Step 1: Get tool call from ClimateGPT
        tool_call_response = chat_with_climategpt(
            system_prompt,
            f"Question: {question}\n\nReturn ONLY ONE valid JSON tool call. Do NOT return multiple JSON objects. No other text, no quotes, no markdown."
        ).strip()
        
        # Clean up the response to ensure it's valid JSON
        tool_call_response = tool_call_response.replace('```json', '').replace('```', '').strip()
        
        # If the model didn't return JSON, try again with a simpler prompt
        if not (tool_call_response.startswith("{") and '"tool"' in tool_call_response):
            tool_call_response = chat_with_climategpt(
                system_prompt,
                f"Question: {question}\n\nReturn ONLY this JSON format: {{\"tool\":\"query\",\"args\":{{\"file_id\":\"transport-country-year\",\"select\":[\"country_name\",\"year\",\"MtCO2\"],\"limit\":10}}}}"
            ).strip()
            tool_call_response = tool_call_response.replace('```json', '').replace('```', '').strip()

        # Apply post-processing constraints (enforce level/where if detected)
        if geo_constraints:
            tool_call_response = apply_level_constraints_to_tool(tool_call_response, geo_constraints)

        used_tool = "unknown"
        try:
            json_start = tool_call_response.find("{")
            json_end = tool_call_response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_candidate = tool_call_response[json_start:json_end]
                tool_call_obj = json.loads(json_candidate)
                used_tool = tool_call_obj.get("tool") or tool_call_obj.get("tool_call") or "unknown"
                logger.info(f"Extracted tool: {used_tool} from tool call")
            else:
                logger.warning(f"No JSON found in response: {tool_call_response[:200]}...")
        except Exception as parse_error:
            logger.warning(f"Failed to parse tool call response: {parse_error}, response: {tool_call_response[:200]}...")
            used_tool = "unknown"

        # Step 2: Execute tool call
        result = exec_tool_call(tool_call_response)

        # Step 3: Handle tool call errors with fallback responses
        if "error" in result:
            error_msg = result["error"]
            if "circuit breaker" in error_msg.lower():
                return create_fallback_response(question, {"error_type": "circuit_breaker"}), {}, used_tool
            elif "timeout" in error_msg.lower():
                return create_fallback_response(question, {"error_type": "timeout"}), {}, used_tool
            elif "connection" in error_msg.lower():
                return create_fallback_response(question, {"error_type": "network_error"}), {}, used_tool
            else:
                return f"I encountered an error while processing your question: {error_msg}", {}, used_tool

        # Step 3.5: Handle batch query aggregation if present
        if isinstance(result, dict) and "results" in result:
            batch_results = result.get("results", [])
            successful_results = [
                r for r in batch_results if r.get("status") == "success" and r.get("data")
            ]

            if not successful_results:
                return create_fallback_response(question, {"error_type": "data_not_found"}), {}, used_tool

            aggregated_rows: List[Dict[str, Any]] = []
            total_row_count = 0
            sources: List[str] = []

            for batch_result in successful_results:
                data = batch_result.get("data", {})
                rows = data.get("rows", [])
                aggregated_rows.extend(rows)
                total_row_count += data.get("row_count", len(rows))
                meta = data.get("meta", {})
                table_id = meta.get("table_id", "unknown")
                if table_id:
                    sources.append(table_id)

            result = {
                "rows": aggregated_rows,
                "row_count": total_row_count,
                "meta": {
                    "table_id": ", ".join(sources) if sources else "aggregated",
                    "source": "EDGAR v2024 (aggregated)",
                    "units": ["tonnes CO2"],
                    "spatial_resolution": "aggregated",
                    "temporal_resolution": "aggregated",
                },
            }

        # Step 3.6: Check if data was actually returned
        if isinstance(result, dict):
            # Check for empty results
            if "rows" in result and (not result["rows"] or len(result["rows"]) == 0):
                return create_fallback_response(question, {"error_type": "data_not_found"}), {}, used_tool

            # Check for null/empty data in rows
            if "rows" in result and result["rows"]:
                has_valid_data = False
                for row in result["rows"]:
                    if isinstance(row, dict) and any(
                        row.get(key) is not None and row.get(key) != "" and row.get(key) != 0
                        for key in ["emissions_tonnes", "MtCO2", "country_name", "year"]
                    ):
                        has_valid_data = True
                        break

                if not has_valid_data:
                    return create_fallback_response(question, {"error_type": "data_not_found"}), {}, used_tool

        # Add MtCO2 column if missing to reduce unit mistakes
        if isinstance(result, dict) and "rows" in result:
            rows = result["rows"]
            for row in rows:
                if "MtCO2" not in row and "emissions_tonnes" in row and isinstance(row["emissions_tonnes"], (int, float)):
                    row["MtCO2"] = row["emissions_tonnes"] / 1e6
        # Store last_result in session for export
        try:
            st.session_state.last_result = result if isinstance(result, dict) else None
        except Exception:
            pass

        # Create enhanced summary prompt with persona context
        rows_preview = json.dumps(result, ensure_ascii=False)

        # Ensure persona object is available
        persona = PERSONAS.get(persona_key, PERSONAS[CLIMATE_ANALYST])

        tool_description = {
            "query": "single dataset query",
            "batch_query": "aggregated multi-sector query",
            "metrics.yoy": "year-over-year analysis",
            "metrics.rankings": "ranking analysis",
            "metrics.trends": "trend analysis",
        }.get(used_tool, f"{used_tool} tool")

        summary_prompt = f"""
        You are responding as a {persona['name']} with expertise in {persona['expertise']}.
        Your tone should be {persona['tone']} and your response style should be {persona['response_style']}.

        User asked: {question}

        Using the JSON data below (which includes rows and meta), write a comprehensive answer that reflects your persona:
        - If present, compute/compare as needed (YoY deltas, rankings, trends).
        - Always cite: "Source: {result.get('meta',{}).get('table_id','?')}, EDGAR v2024 transport."
        - Use correct units (tonnes CO₂; optionally show MtCO₂ for big numbers).
        - Include insights relevant to your persona's expertise area.
        - Keep the answer informative but concise (4–8 sentences).
        - Maintain your persona's tone and perspective throughout.
        - IMPORTANT: Include at the end: "Data retrieved using MCP {tool_description}."

        Data:
        {rows_preview}
        """

        # Step 4: Get natural language summary with retry-on-empty
        def _shorten_prompt(p: str) -> str:
            # Remove some guidance bullets to reduce token size on retry
            lines = [ln for ln in p.split('\n') if not ln.strip().startswith('- ')]
            return '\n'.join(lines)

        raw_answer = ""
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                prompt_to_use = summary_prompt if attempt == 0 else _shorten_prompt(summary_prompt)
                raw_answer = chat_with_climategpt(system_prompt, prompt_to_use, temperature=0.2)
                # Consider success if non-empty and not the generic error string
                if raw_answer and not raw_answer.strip().lower().startswith("error communicating"):
                    break
            except Exception as e:
                last_error = e
            # Backoff with jitter
            time.sleep(0.5 * (2 ** attempt))

        if not raw_answer or not raw_answer.strip() or raw_answer.strip().lower().startswith("error communicating"):
            if last_error:
                logger.error(f"LLM retries failed with error: {last_error}")
            return create_fallback_response(question, {"error_type": "llm_error"}), {}, used_tool

        # Step 5: Curate the response based on persona with error handling
        try:
            curated_answer = output_controller.curate_response(raw_answer, persona_key, result)
        except Exception as e:
            logger.error(f"Error in response curation: {e}")
            curated_answer = raw_answer  # Fall back to raw answer

        # Debug: Check if answer is empty
        if not curated_answer or not curated_answer.strip():
            return create_fallback_response(question, {"error_type": "empty_response"}), {}, used_tool

        return curated_answer, result, used_tool

    except Exception as e:
        logger.error(f"Unexpected error in process_climategpt_question: {e}")
        return create_fallback_response(question, {"error_type": "unexpected_error"}), {}, "unknown"

# Status check functions
def test_mcp_connection():
    try:
        response = requests.get(f"{MCP_URL}/health", timeout=5)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout, 
              requests.exceptions.ConnectionError, json.JSONDecodeError) as e:
        logger.error(f"Connection test failed: {e}")
        return False

def test_climategpt_connection():
    try:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.2
        }
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={"Content-Type": CONTENT_TYPE_JSON},
            data=json.dumps(payload),
            auth=HTTPBasicAuth(USER, PASS) if USER else None,
            timeout=10,
        )
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout, 
              requests.exceptions.ConnectionError, json.JSONDecodeError) as e:
        logger.error(f"Connection test failed: {e}")
        return False

# Main UI
st.title("🌍 ClimateGPT for Climate Analysts — Multi-Sector Emissions QA")
st.write("Select your professional role and ask questions in plain English. ClimateGPT will provide responses tailored to your analytical needs.")

# Note about caching being disabled for testing
st.info("🚀 **Testing Mode**: Caching disabled for reliable testing and development")

## (Persona selection UI moved inline with the message bar; previous persona grid removed)

# Compact status row
m_ok = test_mcp_connection()
l_ok = test_climategpt_connection()
cb_state = mcp_circuit_breaker.state

status_chip = (
    ("🟢 Healthy" if cb_state == "CLOSED" else ("🟡 Recovering" if cb_state == "HALF_OPEN" else "🔴 Overloaded"))
)
st.caption(
    f"Status: {'🟢' if m_ok else '🔴'} MCP · {'🟢' if l_ok else '🔴'} LLM · {status_chip}"
)

## (Debug & Data Troubleshooting removed per request)

## (Example Questions removed per request)

## (Data Coverage removed per request)

# Conversation and performance management
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    # Export: prefer CSV of last_result rows if available, else conversation JSON
    has_rows = isinstance(st.session_state.get("last_result"), dict) and st.session_state.last_result.get("rows")
    if has_rows:
        try:
            df = pd.DataFrame(st.session_state.last_result.get("rows", []))
            if not df.empty:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Last Result (CSV)",
                    data=csv_bytes,
                    file_name=f"climategpt_result_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        except Exception:
            pass
    if st.button("📊 Export Chat"):
        if st.session_state.messages:
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "selected_persona": st.session_state.selected_persona,
                "conversation": st.session_state.messages
            }
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"climategpt_persona_conversation_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime=CONTENT_TYPE_JSON
            )
        else:
            st.warning("No conversation to export")

with col3:
    if st.button("🔄 Refresh App"):
        st.rerun()

with col4:
    st.caption(f"💬 {len(st.session_state.messages)} messages", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add export button for assistant messages with data
        if message["role"] == "assistant" and "Source:" in message["content"]:
            # Try to extract data from the message for export
            if st.button("📥 Export Data", key="export_" + str(len(st.session_state.messages))):
                # This would need to be enhanced to extract actual data
                st.info("Data export feature - would extract tabular data from response")

st.divider()
c_msg, c_persona = st.columns([0.8, 0.2])

with c_persona:
    persona_options = list(PERSONAS.keys())
    current_idx = persona_options.index(st.session_state.selected_persona) if st.session_state.selected_persona in persona_options else 0
    new_persona = st.selectbox(
        "Persona",
        options=persona_options,
        index=current_idx,
        format_func=lambda k: f"{PERSONAS[k]['icon']} {PERSONAS[k]['name']}",
        label_visibility="collapsed",
        key="persona_inline_select",
    )
    if new_persona != st.session_state.selected_persona:
        st.session_state.selected_persona = new_persona
        try:
            # Streamlit modern API for query params
            try:
                st.query_params["persona"] = new_persona
            except Exception:
                st.query_params.from_dict({"persona": new_persona})
        except Exception:
            pass

with c_msg:
    try:
        qp = st.query_params
        qp_persona = qp.get("persona")
        if isinstance(qp_persona, list):
            qp_persona = qp_persona[0] if qp_persona else None
        if qp_persona and qp_persona in PERSONAS and qp_persona != st.session_state.selected_persona:
            st.session_state.selected_persona = qp_persona
    except Exception:
        pass
    placeholder = f"Ask {PERSONAS[st.session_state.selected_persona]['name']} about climate emissions data..."
    prompt = st.chat_input(placeholder)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(f"{PERSONAS[st.session_state.selected_persona]['name']} is analyzing your question..."):
            try:
                start_time = time.time()
                answer, data_context, used_tool = process_climategpt_question(prompt, st.session_state.selected_persona)
                response_time = time.time() - start_time

                st.markdown(answer)

                if data_context:
                    try:
                        evaluator = ResponseEvaluator()
                        evaluation = evaluator.evaluate_response(
                            question=prompt,
                            response=answer,
                            data_context=data_context,
                            tool_used=used_tool,
                        )

                        with st.expander("📊 Response Quality Metrics", expanded=False):
                            col1, col2, col3 = st.columns(3)

                            accuracy_data = evaluation.get("accuracy", {}) if isinstance(evaluation, dict) else {}
                            groundedness_data = evaluation.get("groundedness", {}) if isinstance(evaluation, dict) else {}
                            completeness_data = evaluation.get("completeness", {}) if isinstance(evaluation, dict) else {}

                            accuracy_score = accuracy_data.get("score", 0) if isinstance(accuracy_data, dict) else 0
                            groundedness_score = groundedness_data.get("score", 0) if isinstance(groundedness_data, dict) else 0
                            completeness_score = completeness_data.get("score", 0) if isinstance(completeness_data, dict) else 0

                            with col1:
                                st.metric("Accuracy", f"{accuracy_score:.2f}", help="Verifies numerical claims against actual data")
                            with col2:
                                st.metric("Groundedness", f"{groundedness_score:.2f}", help="Checks if response references available data points")
                            with col3:
                                st.metric("Completeness", f"{completeness_score:.2f}", help="Ensures all question aspects are addressed")

                            overall_score = evaluation.get("overall_score", 0) if isinstance(evaluation, dict) else 0
                            grade = evaluation.get("grade", "F") if isinstance(evaluation, dict) else "F"
                            st.markdown(f"**Overall Grade: {grade} ({overall_score:.2f})**")

                            if st.checkbox("Show detailed evaluation", key=f"details_{len(st.session_state.messages)}"):
                                st.markdown("### Detailed Evaluation Report")
                                evaluator.print_evaluation_report(evaluation)

                    except Exception as eval_error:
                        st.warning(f"Could not evaluate response: {str(eval_error)}")

                st.caption(f"⏱️ Response time: {response_time:.2f}s | Persona: {PERSONAS[st.session_state.selected_persona]['name']}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
