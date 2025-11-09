#!/usr/bin/env python3
"""
Extract ground truth data for test questions from DuckDB.

This script queries the DuckDB database to extract actual emissions data
for each test question, providing ground truth values for accuracy validation.

Usage:
    python extract_ground_truth.py
"""

import json
import duckdb
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "data/warehouse/climategpt.duckdb"
MANIFEST_PATH = "data/curated-2/manifest_mcp_duckdb.json"
QUESTION_BANK_PATH = "test_question_bank.json"
OUTPUT_PATH = "ground_truth_data.json"


def load_manifest():
    """Load the manifest to understand dataset structure."""
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)


def load_questions():
    """Load the test question bank."""
    with open(QUESTION_BANK_PATH, 'r') as f:
        data = json.load(f)
        return data['questions']


def build_file_id(sector: str, level: str, grain: str) -> str:
    """Build file_id from sector, level, and grain."""
    # Handle spaces and special characters
    sector_clean = sector.replace(' ', '-').lower()
    level_clean = level.replace(' ', '-').lower()
    grain_clean = grain.replace(' ', '-').lower()
    return f"{sector_clean}-{level_clean}-{grain_clean}"


def normalize_country(country: str) -> str:
    """Normalize country names to match database."""
    mapping = {
        "USA": "United States of America",
        "US": "United States of America",
        "UK": "United Kingdom",
        "South Korea": "Republic of Korea",
    }
    return mapping.get(country, country)


def extract_ground_truth_simple(conn, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract ground truth for simple factual questions.

    Example: "What were Germany's transportation sector emissions in 2023?"
    """
    try:
        expected = question.get('expected_structure', {})
        sector = question.get('sector')
        level = question.get('level')
        grain = question.get('grain')

        # Skip multi-sector or complex queries for now
        if sector == 'multiple':
            logger.info(f"Q{question['id']}: Skipping complex multi-sector query")
            return {"note": "Complex query - manual validation required"}

        # Build file_id
        file_id = build_file_id(sector, level, grain)

        # Extract location and year
        location = expected.get('location')
        year = expected.get('year')

        if not location or not year:
            logger.warning(f"Q{question['id']}: Missing location or year in expected_structure")
            return None

        # Normalize location
        location = normalize_country(location)

        # Determine location column based on level
        if level == 'country':
            location_col = 'country_name'
        elif level == 'admin1':
            location_col = 'admin1_name'
        elif level == 'city':
            location_col = 'city_name'
        else:
            logger.warning(f"Q{question['id']}: Unknown level {level}")
            return None

        # Build query based on grain
        if grain == 'year':
            query = f"""
                SELECT
                    {location_col} as location,
                    year,
                    SUM(emissions_quantity) as total_emissions,
                    COUNT(*) as record_count
                FROM read_parquet('data/warehouse/{file_id}.parquet')
                WHERE {location_col} = ? AND year = ?
                GROUP BY {location_col}, year
            """
            params = [location, year]
        elif grain == 'month':
            query = f"""
                SELECT
                    {location_col} as location,
                    year,
                    month,
                    SUM(emissions_quantity) as total_emissions,
                    COUNT(*) as record_count
                FROM read_parquet('data/warehouse/{file_id}.parquet')
                WHERE {location_col} = ? AND year = ?
                GROUP BY {location_col}, year, month
                ORDER BY month
            """
            params = [location, year]
        else:
            logger.warning(f"Q{question['id']}: Unknown grain {grain}")
            return None

        # Execute query
        result = conn.execute(query, params).fetchall()

        if not result:
            logger.warning(f"Q{question['id']}: No data found for {location} {year} in {file_id}")
            return {
                "file_id": file_id,
                "location": location,
                "year": year,
                "found": False,
                "note": "No data available"
            }

        # Format result
        if grain == 'year':
            row = result[0]
            # Convert from kt to Mt (divide by 1000)
            emissions_mt = row[2] / 1000.0 if row[2] else None
            return {
                "file_id": file_id,
                "location": row[0],
                "year": row[1],
                "emissions_mt": round(emissions_mt, 2) if emissions_mt else None,
                "record_count": row[3],
                "found": True
            }
        else:  # month
            monthly_data = []
            for row in result:
                emissions_mt = row[3] / 1000.0 if row[3] else None
                monthly_data.append({
                    "month": row[2],
                    "emissions_mt": round(emissions_mt, 2) if emissions_mt else None
                })

            total_emissions = sum(m['emissions_mt'] for m in monthly_data if m['emissions_mt'])
            return {
                "file_id": file_id,
                "location": result[0][0],
                "year": result[0][1],
                "monthly_data": monthly_data,
                "total_emissions_mt": round(total_emissions, 2),
                "found": True
            }

    except Exception as e:
        logger.error(f"Q{question['id']}: Error extracting ground truth: {e}")
        return {"error": str(e)}


def extract_all_ground_truth():
    """Extract ground truth for all test questions."""
    logger.info("Loading question bank...")
    questions = load_questions()

    logger.info(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH, read_only=True)

    results = []
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/50: Q{question['id']}")

        ground_truth = None
        category = question.get('category')

        if category in ['simple', 'temporal']:
            ground_truth = extract_ground_truth_simple(conn, question)
        elif category == 'comparative':
            # For comparative queries, need to handle multiple locations
            logger.info(f"Q{question['id']}: Comparative query - manual extraction needed")
            ground_truth = {"note": "Comparative query - manual validation required"}
        elif category == 'complex':
            logger.info(f"Q{question['id']}: Complex query - manual extraction needed")
            ground_truth = {"note": "Complex query - manual validation required"}

        results.append({
            "question_id": question['id'],
            "question": question['question'],
            "category": category,
            "sector": question.get('sector'),
            "level": question.get('level'),
            "grain": question.get('grain'),
            "ground_truth": ground_truth
        })

    conn.close()

    # Save results
    logger.info(f"Saving ground truth data to {OUTPUT_PATH}")
    output = {
        "metadata": {
            "generated": "2025-11-02",
            "total_questions": len(results),
            "database": DB_PATH,
            "note": "Comparative and complex queries require manual validation"
        },
        "ground_truth": results
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Ground truth extraction complete!")
    logger.info(f"Results saved to: {OUTPUT_PATH}")

    # Summary
    found_count = sum(1 for r in results if r['ground_truth'] and r['ground_truth'].get('found'))
    logger.info(f"\nSummary:")
    logger.info(f"  Total questions: {len(results)}")
    logger.info(f"  Data found: {found_count}")
    logger.info(f"  Manual validation needed: {len(results) - found_count}")


if __name__ == "__main__":
    # Check if files exist
    if not Path(DB_PATH).exists():
        logger.error(f"Database not found: {DB_PATH}")
        logger.error("This script expects DuckDB warehouse, not just parquet files.")
        logger.error("You may need to adjust the query to use parquet files directly.")
        exit(1)

    if not Path(QUESTION_BANK_PATH).exists():
        logger.error(f"Question bank not found: {QUESTION_BANK_PATH}")
        exit(1)

    extract_all_ground_truth()
