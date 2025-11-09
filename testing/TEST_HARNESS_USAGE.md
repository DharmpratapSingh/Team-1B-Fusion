# Test Harness Usage Guide

## Overview

The automated test harness allows you to run comparative testing between ClimateGPT and Meta Llama (via LM Studio) using a comprehensive question bank.

## Files

- **test_harness.py** - Main test script
- **analyze_results.py** - Results analysis script
- **test_config.json** - Configuration file
- **test_question_bank.json** - Question bank (50 questions)
- **requirements_testing.txt** - Python dependencies

---

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements_testing.txt

# Or using uv
uv pip install -r requirements_testing.txt
```

### 2. Start ClimateGPT Server

```bash
# Terminal 1: Start ClimateGPT
make serve

# Or manually
uv run uvicorn mcp_server:app --host 127.0.0.1 --port 8010
```

### 3. Start LM Studio (for Llama testing)

1. Download and install LM Studio: https://lmstudio.ai/
2. Open LM Studio
3. Download a model (recommended: Meta-Llama-3.1-8B-Instruct)
4. Go to "Local Server" tab
5. Click "Start Server"
6. Verify it's running at http://localhost:1234

### 4. Run Tests

```bash
# Test both systems (ClimateGPT + Llama)
python test_harness.py

# Test only ClimateGPT
python test_harness.py --climategpt-only

# Run pilot test (10 questions)
python test_harness.py --pilot
```

---

## Test Harness Options

### Basic Usage

```bash
# Test both systems with all 50 questions
python test_harness.py

# Test only ClimateGPT
python test_harness.py --climategpt-only

# Test only Llama
python test_harness.py --llama-only
```

### Selective Testing

```bash
# Run pilot test (10 representative questions)
python test_harness.py --pilot

# Test specific questions
python test_harness.py --questions 1,2,3,4,5

# Test first 5 questions from each sector
python test_harness.py --questions 1,7,13,19,25,31,37,43
```

### Configuration

```bash
# Use custom config file
python test_harness.py --config my_config.json

# Enable verbose logging
python test_harness.py --verbose
```

### Combined Options

```bash
# Pilot test with ClimateGPT only
python test_harness.py --pilot --climategpt-only

# Specific questions with verbose output
python test_harness.py --questions 1,2,3 --verbose
```

---

## Configuration File (test_config.json)

### Default Configuration

```json
{
  "climategpt": {
    "url": "http://localhost:8010",
    "endpoint": "/query",
    "timeout": 30
  },
  "llama": {
    "url": "http://localhost:1234",
    "endpoint": "/v1/chat/completions",
    "model": "meta-llama-3.1-8b-instruct",
    "timeout": 30,
    "temperature": 0.1,
    "max_tokens": 500,
    "system_prompt": "You are an expert on climate emissions data..."
  },
  "test": {
    "question_bank": "test_question_bank.json",
    "output_dir": "test_results",
    "delay_between_requests": 1.0,
    "max_retries": 2,
    "retry_delay": 2.0
  }
}
```

### Customization Options

| Section | Option | Description | Default |
|---------|--------|-------------|---------|
| climategpt.url | URL | ClimateGPT server URL | http://localhost:8010 |
| climategpt.timeout | Seconds | Request timeout | 30 |
| llama.url | URL | LM Studio server URL | http://localhost:1234 |
| llama.model | String | Model identifier | meta-llama-3.1-8b-instruct |
| llama.temperature | Float | Response randomness (0-1) | 0.1 |
| llama.max_tokens | Integer | Max response length | 500 |
| test.delay_between_requests | Seconds | Delay between API calls | 1.0 |
| test.max_retries | Integer | Retry attempts on failure | 2 |

---

## Output Files

### JSON Results

**Location**: `test_results/test_results_YYYYMMDD_HHMMSS.json`

**Structure**:
```json
{
  "metadata": {
    "test_date": "2025-11-02T14:30:22",
    "total_questions": 50,
    "total_tests": 100,
    "config": {...}
  },
  "results": [
    {
      "question_id": 1,
      "question": "What were Germany's transportation...",
      "category": "simple",
      "sector": "transport",
      "level": "country",
      "grain": "year",
      "difficulty": "easy",
      "results": {
        "climategpt": {
          "response": "Germany's transportation sector...",
          "response_time_ms": 1234.56,
          "status_code": 200,
          "error": null,
          "timestamp": "2025-11-02T14:30:23"
        },
        "llama": {
          "response": "I don't have access to...",
          "response_time_ms": 890.12,
          "status_code": 200,
          "error": null,
          "timestamp": "2025-11-02T14:30:25"
        }
      }
    }
  ]
}
```

### CSV Results

**Location**: `test_results/test_results_YYYYMMDD_HHMMSS.csv`

**Columns**:
- question_id
- question
- category
- sector
- level
- grain
- difficulty
- system (climategpt or llama)
- response
- response_time_ms
- status_code
- error
- timestamp

**Use Cases**:
- Open in Excel/Google Sheets for manual review
- Import into analysis tools
- Manual scoring and annotation

---

## Analysis Script

### Basic Analysis

```bash
# Analyze latest results (automatic)
python analyze_results.py

# Analyze specific results file
python analyze_results.py --file test_results/test_results_20251102_143022.json
```

### Generate Visualizations

```bash
# Create charts (requires matplotlib, seaborn)
python analyze_results.py --visualize
```

**Generated Charts**:
1. `response_time_comparison.png` - Box plot of response times
2. `success_rate_by_category.png` - Success rates by question category
3. `response_time_by_sector.png` - Response times by sector
4. `response_length_comparison.png` - Response length comparison

### Export Report

```bash
# Export text report
python analyze_results.py --report
```

**Output**: `test_results/analysis_report_YYYYMMDD_HHMMSS.txt`

### Combined Analysis

```bash
# Full analysis with visualizations and report
python analyze_results.py --visualize --report
```

---

## Example Workflows

### Workflow 1: Quick Pilot Test

```bash
# 1. Start services
make serve &                                    # Start ClimateGPT
# Start LM Studio manually

# 2. Run pilot test (10 questions)
python test_harness.py --pilot

# 3. Analyze results
python analyze_results.py

# Output:
# - test_results/test_results_20251102_143022.json
# - test_results/test_results_20251102_143022.csv
# - Console summary with statistics
```

**Time**: ~5-10 minutes

### Workflow 2: Full Test with Analysis

```bash
# 1. Start services
make serve &
# Start LM Studio

# 2. Run full test (50 questions)
python test_harness.py

# 3. Generate full analysis with charts
python analyze_results.py --visualize --report

# Output:
# - test_results/test_results_20251102_150022.json
# - test_results/test_results_20251102_150022.csv
# - test_results/analysis_report_20251102_150200.txt
# - test_results/*.png (4 charts)
```

**Time**: ~15-20 minutes

### Workflow 3: Test Only ClimateGPT

```bash
# Useful when LM Studio is not available or you want to validate ClimateGPT only

# 1. Start ClimateGPT
make serve

# 2. Test ClimateGPT only
python test_harness.py --climategpt-only

# 3. Analyze
python analyze_results.py
```

**Time**: ~5-10 minutes

### Workflow 4: Test Specific Questions

```bash
# Test questions about power sector (Q7-Q12)
python test_harness.py --questions 7,8,9,10,11,12

# Test complex questions only (Q48-Q50)
python test_harness.py --questions 48,49,50

# Test monthly grain questions
python test_harness.py --questions 4,10,16,22,28,33,40,46,50
```

---

## Troubleshooting

### Service Check Failures

**Error**: `✗ ClimateGPT is not reachable at http://localhost:8010`

**Solutions**:
```bash
# Check if server is running
lsof -i :8010

# Start the server
make serve

# Or manually
uv run uvicorn mcp_server:app --host 127.0.0.1 --port 8010
```

**Error**: `✗ LM Studio is not reachable at http://localhost:1234`

**Solutions**:
1. Open LM Studio
2. Download a model if not already done
3. Go to "Local Server" tab
4. Click "Start Server"
5. Verify model is loaded

### Timeout Errors

**Error**: `Timeout after 30s`

**Solutions**:
1. Increase timeout in `test_config.json`:
   ```json
   {
     "climategpt": {
       "timeout": 60
     }
   }
   ```
2. Check server performance
3. Reduce concurrent load

### Connection Errors

**Error**: `Connection refused`

**Check**:
```bash
# Verify ClimateGPT
curl http://localhost:8010/health

# Verify LM Studio
curl http://localhost:1234/v1/models
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'requests'`

**Solution**:
```bash
# Install dependencies
pip install -r requirements_testing.txt

# Or
uv pip install requests pandas matplotlib seaborn
```

### File Not Found Errors

**Error**: `Question bank not found: test_question_bank.json`

**Solution**:
```bash
# Run from project root
cd /Users/dharmpratapsingh/Downloads/DataSets_ClimateGPT

# Or specify full path in config
{
  "test": {
    "question_bank": "/full/path/to/test_question_bank.json"
  }
}
```

---

## Understanding Results

### Success Metrics

**Good ClimateGPT Performance**:
- Success rate: >95%
- Response time: <2000ms average
- Errors: <5%

**Expected Llama Performance**:
- Success rate: ~100% (but wrong data)
- Response time: 500-1500ms average
- Likely to say "I don't have access to real-time data"

### Response Analysis

When analyzing responses, look for:

**ClimateGPT**:
- ✓ Specific numerical data (e.g., "245.3 MtCO2")
- ✓ Correct units (MtCO2)
- ✓ Location and year matching query
- ✓ Source attribution (EDGAR v2024)
- ❓ Response formatting quality
- ❓ Error message clarity

**Llama**:
- ✗ Generic responses or disclaimers
- ✗ Hallucinated numbers (incorrect data)
- ✓ Natural language quality
- ✓ Conversational tone
- ❓ Attempts to be helpful despite no data

---

## Next Steps After Testing

### 1. Manual Scoring

Add manual scores to CSV:
```csv
question_id,climategpt_accuracy,climategpt_completeness,llama_accuracy,llama_completeness
1,5,4,1,3
2,5,5,1,2
```

Use 1-5 scale for:
- **Accuracy**: Correctness of data
- **Completeness**: Answers all parts
- **Quality**: Formatting and clarity
- **Usefulness**: Actionable information

### 2. Identify Improvements

Based on results, create improvement list:
1. Response formatting enhancements
2. Better error messages
3. Improved fallback explanations
4. Natural language quality improvements
5. User experience refinements

### 3. Implement Changes

Focus on top 3-5 weaknesses:
```bash
# Example: Improve answer formatting
# Edit: src/utils/answer.py

# Re-test specific questions
python test_harness.py --questions 1,2,3,4,5

# Compare results
python analyze_results.py
```

### 4. Iterate

Repeat testing cycle:
1. Test → 2. Analyze → 3. Improve → 4. Re-test

---

## Advanced Usage

### Custom System Prompts

Test different prompts for Llama:

```json
{
  "llama": {
    "system_prompt": "You are a climate scientist specializing in emissions data. Always cite sources and be precise with numbers."
  }
}
```

### Multiple Models

Test different Llama versions:

```bash
# Test Llama 3.1 8B
python test_harness.py --config config_llama31_8b.json

# Test Llama 3.2 3B
python test_harness.py --config config_llama32_3b.json

# Compare results
python analyze_results.py --file test_results/test_results_llama31.json
python analyze_results.py --file test_results/test_results_llama32.json
```

### Batch Testing

Test multiple configurations:

```bash
#!/bin/bash
# batch_test.sh

configs=("config1.json" "config2.json" "config3.json")

for config in "${configs[@]}"; do
  echo "Testing with $config"
  python test_harness.py --config "$config"
  sleep 5
done

echo "All tests complete!"
```

---

## Tips and Best Practices

### Before Testing

1. ✓ Verify both services are running
2. ✓ Test with pilot first (10 questions)
3. ✓ Check disk space for results
4. ✓ Close other heavy applications

### During Testing

1. ✓ Monitor console output for errors
2. ✓ Don't interrupt running tests
3. ✓ Watch for timeouts (may need config adjustment)
4. ✓ Results save automatically

### After Testing

1. ✓ Review console summary first
2. ✓ Check for errors in analysis
3. ✓ Generate visualizations for patterns
4. ✓ Export report for documentation
5. ✓ Keep raw JSON for reference

### Data Management

```bash
# Organize results by date
mkdir -p test_results/2025-11-02
mv test_results/test_results_20251102_*.* test_results/2025-11-02/

# Archive old results
tar -czf test_results_archive_20251102.tar.gz test_results/2025-11-02/
```

---

## FAQ

**Q: How long does the full test take?**
A: ~15-20 minutes for 50 questions × 2 systems with 1s delay between requests.

**Q: Can I test in parallel?**
A: Not recommended. Sequential testing prevents rate limiting and server overload.

**Q: What if ClimateGPT returns an error?**
A: The test continues. Errors are logged and included in results for analysis.

**Q: Can I test other LLMs?**
A: Yes! Modify the config to point to different API endpoints (OpenRouter, Together AI, etc.).

**Q: How do I compare multiple test runs?**
A: Keep the JSON files and use analyze_results.py with different --file options.

**Q: What's the difference between pilot and full test?**
A: Pilot uses 10 representative questions (fast), full uses all 50 (comprehensive).

---

## Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Verify service status
4. Check configuration file
5. Review sample output files

---

**Last Updated**: 2025-11-02
**Version**: 1.0
