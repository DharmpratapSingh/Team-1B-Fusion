# Quick Start - Comparative LLM Testing

Get testing in 5 minutes! ⚡

---

## Prerequisites

- ✅ ClimateGPT installed (parent directory)
- ✅ LM Studio installed with Meta Llama model
- ✅ Python 3.9+

---

## Step 1: Install Dependencies (1 minute)

```bash
cd testing
pip install -r requirements_testing.txt
```

---

## Step 2: Verify Setup (30 seconds)

```bash
python verify_setup.py
```

Expected output:
```
================================================================================
✅ SETUP VERIFICATION PASSED!

You're ready to run tests!
================================================================================
```

---

## Step 3: Start ClimateGPT (Terminal 1)

```bash
cd ..
make serve
```

Wait for:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8010
```

---

## Step 4: Start LM Studio

1. Open **LM Studio** app
2. Go to **"Local Server"** tab
3. Verify model is loaded: `meta-llama-3.1-8b-instruct@q4_k_m`
4. Click **"Start Server"**
5. Verify: http://localhost:1234

Quick test:
```bash
curl http://localhost:1234/v1/models
```

Should show your loaded model.

---

## Step 5: Run Pilot Test (Terminal 2)

```bash
cd testing
python test_harness.py --pilot
```

Expected output:
```
Checking services...
  ✓ ClimateGPT is running at http://localhost:8010
  ✓ LM Studio is running at http://localhost:1234

Starting tests: 10 questions × 2 systems = 20 tests
────────────────────────────────────────────────────────────────

[1/10] Q1: What were Germany's transportation sector emissions...
  → Testing ClimateGPT...
    ✓ Response: 1234 chars, 1234.56ms
  → Testing Llama...
    ✓ Response: 456 chars, 890.12ms

[2/10] Q7: What were the power sector emissions for France...
  → Testing ClimateGPT...
    ✓ Response: 1567 chars, 1456.78ms
  → Testing Llama...
    ✓ Response: 523 chars, 923.45ms

...

Testing complete! Total time: 45.2s
Total tests run: 20
Success: 20, Errors: 0

Results saved to: test_results/test_results_20251102_143022.json
CSV results saved to: test_results/test_results_20251102_143022.csv
```

---

## Step 6: Analyze Results

```bash
python analyze_results.py
```

Expected output:
```
================================================================================
TEST RESULTS SUMMARY
================================================================================

CLIMATEGPT:
  Total tests: 10
  Successful: 10 (100.0%)
  Response time: Mean 1234.5ms, Median 1150.0ms

LLAMA:
  Total tests: 10
  Successful: 10 (100.0%)
  Response time: Mean 890.2ms, Median 850.0ms

────────────────────────────────────────────────────────────────
PERFORMANCE BY CATEGORY
────────────────────────────────────────────────────────────────

SIMPLE:
  climategpt: 6/6 success, 1123.4ms avg
  llama: 6/6 success, 856.7ms avg

...
```

---

## 🎉 You're Done!

Results are saved in:
- `test_results/test_results_YYYYMMDD_HHMMSS.json`
- `test_results/test_results_YYYYMMDD_HHMMSS.csv`

---

## Next Steps

### Option 1: Full Test (All 50 Questions)

```bash
python test_harness.py
python analyze_results.py --visualize --report
```

**Time**: ~20 minutes
**Output**: Complete analysis with charts

### Option 2: Review Results

```bash
# Open CSV in spreadsheet
open test_results/test_results_*.csv

# Or view JSON
cat test_results/test_results_*.json | jq .
```

### Option 3: Test Specific Areas

```bash
# Test only monthly data questions
python test_harness.py --questions 4,10,16,22,28,33,40,46,50

# Test only comparative queries
python test_harness.py --questions 5,11,17,23,34,47,49

# Test only complex questions
python test_harness.py --questions 48,49,50
```

---

## Troubleshooting

### ✗ ClimateGPT Not Reachable

```bash
# Check if running
lsof -i :8010

# Start it (in parent directory)
cd ..
make serve
```

### ✗ LM Studio Not Reachable

```bash
# Check if responding
curl http://localhost:1234/v1/models

# If not:
# 1. Open LM Studio
# 2. Load model
# 3. Start server
```

### Import Errors

```bash
pip install -r requirements_testing.txt
```

### Wrong Model ID

Your model ID from LM Studio is:
```
meta-llama-3.1-8b-instruct@q4_k_m
```

This is already configured in `test_config.json`. No changes needed!

If you want to use Q5_K_M instead:
```bash
# Edit test_config.json
nano test_config.json

# Change line 10 to:
"model": "meta-llama-3.1-8b-instruct@q5_k_m",
```

---

## Common Commands

```bash
# Verify setup
python verify_setup.py

# Pilot test (10 questions)
python test_harness.py --pilot

# Full test (50 questions)
python test_harness.py

# Test only ClimateGPT
python test_harness.py --climategpt-only

# Analyze latest results
python analyze_results.py

# Generate charts
python analyze_results.py --visualize

# Export report
python analyze_results.py --report
```

---

## What You'll Learn

From the pilot test you'll discover:

1. **Accuracy Gap**: How much more accurate is ClimateGPT?
2. **Response Speed**: Which is faster?
3. **Response Quality**: Which has better formatting?
4. **Error Handling**: How well does each handle edge cases?
5. **User Experience**: Which provides better UX?

**Expected**: ClimateGPT will have much higher accuracy but you'll identify areas to improve formatting and UX.

---

## Time Estimates

| Task | Duration |
|------|----------|
| Install deps | 1 min |
| Verify setup | 30 sec |
| Start services | 1 min |
| Pilot test | 2-3 min |
| Analysis | 1 min |
| **Total** | **~5-7 min** |

---

## What's Next?

After pilot test:

1. ✅ Review sample responses
2. ✅ Check accuracy differences
3. ✅ Identify ClimateGPT weaknesses
4. ✅ Run full test (optional)
5. ✅ Implement improvements

---

**Ready?** Run these 3 commands:

```bash
# 1. Verify
python verify_setup.py

# 2. Test (after starting ClimateGPT and LM Studio)
python test_harness.py --pilot

# 3. Analyze
python analyze_results.py
```

**🚀 Good luck!**
