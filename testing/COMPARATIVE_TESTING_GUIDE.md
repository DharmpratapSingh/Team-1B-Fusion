# Comparative LLM Testing Guide

## Overview

This guide provides instructions for testing ClimateGPT against other LLMs (primarily Meta Llama) using a comprehensive question bank of 50 questions.

## Question Bank Coverage

### Complete Coverage Summary

| Dimension | Coverage |
|-----------|----------|
| **Total Questions** | 50 |
| **Sectors** | 8 (all sectors covered) |
| **Geographic Levels** | 3 (country, admin1, city) |
| **Temporal Grains** | 2 (year, month) |
| **Difficulty Levels** | 3 (easy, medium, hard) |

### Distribution Details

#### By Sector (All 8 Covered)
- Transport: 6 questions
- Power: 6 questions
- Waste: 6 questions
- Agriculture: 6 questions
- Buildings: 6 questions
- Fuel Exploitation: 6 questions
- Industrial Combustion: 6 questions
- Industrial Processes: 5 questions
- Multiple Sectors: 3 questions

#### By Geographic Level
- Country: 28 questions (56%)
- Admin1 (States/Provinces): 13 questions (26%)
- City: 9 questions (18%)

#### By Temporal Grain
- Annual (year): 38 questions (76%)
- Monthly: 12 questions (24%)

#### By Category
- Simple (factual queries): 26 questions (52%)
- Temporal (trends, YoY): 14 questions (28%)
- Comparative (multi-location): 7 questions (14%)
- Complex (multi-sector): 3 questions (6%)

#### By Difficulty
- Easy: 26 questions (52%)
- Medium: 21 questions (42%)
- Hard: 3 questions (6%)

---

## Testing Setup

### Prerequisites

1. **ClimateGPT Running**
   ```bash
   # Ensure ClimateGPT server is running
   cd /Users/dharmpratapsingh/Downloads/DataSets_ClimateGPT
   make serve
   # Or manually:
   uv run uvicorn mcp_server:app --host 127.0.0.1 --port 8010
   ```

2. **LM Studio Setup (for Meta Llama)**
   - Download LM Studio: https://lmstudio.ai/
   - Install and launch LM Studio
   - Download a model (recommended):
     - Meta-Llama-3.1-8B-Instruct (4.7GB)
     - Meta-Llama-3.2-3B-Instruct (lighter, 2GB)
   - Start local server:
     - In LM Studio: Click "Local Server" tab
     - Click "Start Server"
     - Default endpoint: http://localhost:1234/v1

3. **Test Question Bank**
   - JSON version: `test_question_bank.json`
   - CSV version: `test_question_bank.csv`

---

## Testing Methods

### Method 1: Manual Testing (Recommended for Initial Testing)

**Step-by-step:**

1. **Open both systems:**
   - ClimateGPT: http://localhost:8501 (Streamlit UI)
   - LM Studio: Chat interface

2. **For each question:**
   - Ask the same question to both systems
   - Copy responses to results spreadsheet
   - Rate each response (see Scoring section below)

3. **Record results in spreadsheet:**
   - Use template: `test_results_template.csv`

**Pros:**
- Better understanding of responses
- Can evaluate nuance and context
- Easy to start immediately

**Cons:**
- Time-consuming (50 questions × 2 systems)
- Manual data entry prone to errors

---

### Method 2: Automated Testing (Recommended for Full Testing)

**Step 1: Create test harness script**

```python
# test_harness.py
import json
import requests
import time
from datetime import datetime

# Load questions
with open('test_question_bank.json', 'r') as f:
    data = json.load(f)
    questions = data['questions']

# Test ClimateGPT
def test_climategpt(question_text):
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8010/query",
            json={"question": question_text},
            timeout=30
        )
        elapsed = (time.time() - start_time) * 1000
        return {
            "response": response.json(),
            "response_time_ms": elapsed,
            "status_code": response.status_code,
            "error": None
        }
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return {
            "response": None,
            "response_time_ms": elapsed,
            "status_code": None,
            "error": str(e)
        }

# Test Llama via LM Studio
def test_llama(question_text):
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "meta-llama-3.1-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert on climate emissions data. Provide accurate, concise answers with specific numbers when possible."
                    },
                    {
                        "role": "user",
                        "content": question_text
                    }
                ],
                "temperature": 0.1,  # Low temp for factual responses
                "max_tokens": 500
            },
            timeout=30
        )
        elapsed = (time.time() - start_time) * 1000
        return {
            "response": response.json()['choices'][0]['message']['content'],
            "response_time_ms": elapsed,
            "status_code": response.status_code,
            "error": None
        }
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return {
            "response": None,
            "response_time_ms": elapsed,
            "status_code": None,
            "error": str(e)
        }

# Run all tests
results = []
for i, q in enumerate(questions, 1):
    print(f"Testing question {i}/50: {q['question'][:50]}...")

    result = {
        "question_id": q['id'],
        "question": q['question'],
        "category": q['category'],
        "sector": q['sector'],
        "level": q['level'],
        "grain": q['grain'],
        "difficulty": q['difficulty'],
        "climategpt": test_climategpt(q['question']),
        "llama": test_llama(q['question']),
        "timestamp": datetime.now().isoformat()
    }

    results.append(result)
    time.sleep(1)  # Rate limiting

# Save results
output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nTesting complete! Results saved to: {output_file}")
```

**Step 2: Run the test harness**

```bash
# Install dependencies if needed
uv pip install requests

# Run tests
python test_harness.py
```

**Pros:**
- Fast execution (5-10 minutes for all 50 questions)
- Consistent testing methodology
- Automatic data collection
- Timestamped results

**Cons:**
- Requires both servers running
- Need to manually score accuracy afterward

---

## Scoring Methodology

### Evaluation Criteria (1-5 scale)

For each question and each LLM, rate on the following dimensions:

#### 1. Accuracy (Most Important)
- **5**: Perfectly accurate data, correct units, valid source
- **4**: Accurate data with minor formatting issues
- **3**: Partially accurate (correct order of magnitude)
- **2**: Incorrect data but relevant attempt
- **1**: Completely wrong or hallucinated data
- **0**: No answer or error

#### 2. Completeness
- **5**: Answers all parts of question with context
- **4**: Answers main question with minor omissions
- **3**: Partial answer, missing some requested info
- **2**: Minimal answer, significant gaps
- **1**: Barely addresses the question
- **0**: No relevant information

#### 3. Response Quality
- **5**: Clear, well-formatted, professional
- **4**: Good quality with minor formatting issues
- **3**: Adequate but could be clearer
- **2**: Poorly formatted or confusing
- **1**: Very poor quality or unclear
- **0**: Unusable response

#### 4. Usefulness
- **5**: Immediately actionable, includes insights
- **4**: Useful with good context
- **3**: Adequate for basic needs
- **2**: Minimally useful
- **1**: Not useful
- **0**: Useless or misleading

---

## Expected Results

### ClimateGPT Expected Performance

**Strengths:**
- Accuracy: 4.5-5.0 (has real data from DuckDB)
- Completeness: 4.0-4.5 (structured tool calls)
- Response Quality: 3.5-4.5 (depends on answer.py formatting)
- Usefulness: 4.0-5.0 (actionable data with sources)

**Potential Weaknesses to Identify:**
- Response formatting (too verbose or too terse?)
- Handling ambiguous questions
- Error messages clarity
- Fallback explanations
- Natural language quality

### Meta Llama Expected Performance

**Strengths:**
- Response Quality: 4.0-4.5 (natural conversational tone)
- Completeness: 3.0-4.0 (tries to be comprehensive)
- Usefulness: 2.0-3.0 (explanatory but not factual)

**Expected Weaknesses:**
- Accuracy: 0-2.0 (no real data access, will hallucinate)
- Will likely say "I don't have access to real-time data"
- May provide outdated or generic information
- Cannot provide specific numerical answers

---

## Sample Questions for Quick Testing (Pilot)

If you want to start with a smaller pilot test, use these 10 questions that cover all major dimensions:

### Pilot Test Set (10 Questions)

1. **Q1** - Simple/Country/Year: "What were Germany's transportation sector emissions in 2023?"
2. **Q7** - Simple/Country/Year: "What were the power sector emissions for France in 2023?"
3. **Q13** - Simple/Country/Year: "What were Japan's waste sector emissions in 2022?"
4. **Q4** - Temporal/Month: "What were USA's monthly transportation emissions for each month in 2023?"
5. **Q5** - Comparative/Country: "Compare transportation emissions between USA and China in 2022."
6. **Q11** - Comparative/Admin1: "Which US state had the highest power sector emissions in 2022?"
7. **Q17** - Comparative/City: "Compare waste emissions between New York City and Los Angeles in 2022."
8. **Q23** - Complex/Ranking: "Rank the top 5 countries by agricultural emissions in 2022."
9. **Q48** - Complex/Multi-sector: "What are the total emissions from all sectors for Germany in 2023?"
10. **Q50** - Complex/Multi-sector/Month: "Compare monthly power and transport emissions for California in 2023."

---

## Results Template

### CSV Template Structure

Create a CSV file with these columns:

```csv
question_id,question,sector,level,grain,difficulty,climategpt_response,climategpt_accuracy,climategpt_completeness,climategpt_quality,climategpt_usefulness,climategpt_time_ms,llama_response,llama_accuracy,llama_completeness,llama_quality,llama_usefulness,llama_time_ms,notes
```

### JSON Results Structure

```json
{
  "test_metadata": {
    "test_date": "2025-11-02",
    "climategpt_version": "mcp_server.py",
    "llama_model": "Meta-Llama-3.1-8B-Instruct",
    "total_questions": 50
  },
  "results": [
    {
      "question_id": 1,
      "question": "What were Germany's transportation sector emissions in 2023?",
      "climategpt": {
        "response": "...",
        "accuracy": 5,
        "completeness": 4,
        "quality": 4,
        "usefulness": 5,
        "response_time_ms": 1200
      },
      "llama": {
        "response": "...",
        "accuracy": 1,
        "completeness": 3,
        "quality": 4,
        "usefulness": 2,
        "response_time_ms": 800
      }
    }
  ]
}
```

---

## Analysis Plan

After collecting results, analyze:

### 1. Aggregate Metrics

```python
import json
import pandas as pd

# Load results
with open('test_results.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame for analysis
results_df = pd.DataFrame(data['results'])

# Calculate averages
metrics = ['accuracy', 'completeness', 'quality', 'usefulness']
for metric in metrics:
    climategpt_avg = results_df[f'climategpt_{metric}'].mean()
    llama_avg = results_df[f'llama_{metric}'].mean()
    print(f"{metric.capitalize()}:")
    print(f"  ClimateGPT: {climategpt_avg:.2f}")
    print(f"  Llama: {llama_avg:.2f}")
    print(f"  Difference: {climategpt_avg - llama_avg:.2f}\n")
```

### 2. Performance by Category

Compare performance across:
- Sectors
- Geographic levels
- Temporal grains
- Difficulty levels
- Question categories

### 3. Identify Weaknesses

Look for patterns where ClimateGPT underperforms:
- Specific question types
- Certain sectors
- Edge cases
- Error handling scenarios

### 4. Improvement Recommendations

Based on findings, create actionable recommendations:
- Prompt improvements
- Response formatting enhancements
- Better error messages
- Enhanced fallback logic

---

## Timeline

### Recommended Schedule

**Day 1: Pilot Testing (5-10 questions)**
- Setup LM Studio
- Test both systems manually
- Validate methodology
- Adjust if needed

**Day 2: Full Automated Testing**
- Run full 50-question test suite
- Collect all responses
- Initial review

**Day 3: Manual Scoring**
- Score all responses
- Calculate metrics
- Identify patterns

**Day 4: Analysis**
- Create comparison charts
- Document findings
- List improvement recommendations

**Day 5: Refinements**
- Implement top 3-5 improvements
- Re-test problem areas
- Document results

---

## Tools and Scripts

### Required Files

1. `test_question_bank.json` - Question bank (created ✓)
2. `test_question_bank.csv` - CSV version (created ✓)
3. `test_harness.py` - Automated testing script (template provided above)
4. `test_results_template.csv` - Results tracking template
5. `analyze_results.py` - Analysis script (create as needed)

### Optional Tools

- Jupyter notebook for interactive analysis
- Matplotlib/Plotly for visualization
- Excel/Google Sheets for manual scoring

---

## Next Steps

1. **Review question bank** - Confirm coverage is adequate
2. **Choose testing method** - Manual pilot or full automated
3. **Setup LM Studio** - Download and configure Llama model
4. **Run pilot test** - Test with 5-10 questions first
5. **Full testing** - Run all 50 questions
6. **Analysis** - Score and analyze results
7. **Refinements** - Implement improvements based on findings

---

## Questions to Answer

After testing, you should be able to answer:

1. **Accuracy Gap**: How much more accurate is ClimateGPT than Llama?
2. **Response Quality**: Is ClimateGPT's formatting competitive?
3. **Weaknesses**: What are ClimateGPT's main weaknesses?
4. **Error Handling**: How well does ClimateGPT handle edge cases?
5. **User Experience**: Which system provides a better UX?
6. **Improvements**: What are the top 5 areas to improve?

---

**Last Updated**: 2025-11-02
**Files**:
- test_question_bank.json (50 questions)
- test_question_bank.csv (spreadsheet version)
- COMPARATIVE_TESTING_GUIDE.md (this file)
