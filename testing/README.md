# Testing Directory

This directory contains all files for comparative LLM testing. After testing is complete, you can easily remove this entire directory to clean up.

---

## 📁 Directory Structure

```
testing/
├── README.md                          # This file
│
├── Core Scripts
├── test_harness.py                    # Main test execution script
├── analyze_results.py                 # Results analysis and visualization
├── verify_setup.py                    # Pre-flight checks
├── extract_ground_truth.py            # Ground truth data extractor
│
├── Configuration
├── test_config.json                   # Test configuration
│
├── Question Bank
├── test_question_bank.json            # 50 test questions
├── test_question_bank.csv             # CSV version
├── test_results_template.csv          # Manual scoring template
│
├── Dependencies
├── requirements_testing.txt           # Python dependencies
│
├── Documentation
├── QUICKSTART.md                      # Quick start guide
├── TEST_HARNESS_USAGE.md              # Complete usage docs
├── COMPARATIVE_TESTING_GUIDE.md       # Testing methodology
├── QUESTION_BANK_SUMMARY.md           # Coverage analysis
├── LM_STUDIO_SETUP.md                 # LM Studio setup guide
│
└── Results (generated during testing)
    └── test_results/                  # Test outputs and analysis
```

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies

```bash
cd testing
pip install -r requirements_testing.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

Expected output:
```
✅ SETUP VERIFICATION PASSED!
```

### 3. Run Pilot Test

**Terminal 1 - Start ClimateGPT:**
```bash
cd ..
make serve
```

**Terminal 2 - Run Test:**
```bash
cd testing
python test_harness.py --pilot
```

---

## 📊 What Gets Tested

- **50 questions** covering all ClimateGPT capabilities
- **8 sectors**: transport, power, waste, agriculture, buildings, fuel-exploitation, industrial-combustion, industrial-processes
- **3 levels**: country, admin1 (states), city
- **2 temporal grains**: yearly, monthly
- **4 question types**: simple, temporal, comparative, complex

---

## 🎯 Testing Goals

1. **Accuracy**: Compare ClimateGPT vs Meta Llama
2. **Performance**: Response time benchmarks
3. **Quality**: Response formatting and clarity
4. **Coverage**: Test all sectors, levels, and grains
5. **Improvements**: Identify areas to enhance

---

## 📚 Documentation Guide

| File | Purpose | Read When |
|------|---------|-----------|
| **QUICKSTART.md** | Quick 5-minute start | First time setup |
| **TEST_HARNESS_USAGE.md** | Complete reference | Running tests |
| **LM_STUDIO_SETUP.md** | LM Studio setup | Installing Llama |
| **COMPARATIVE_TESTING_GUIDE.md** | Testing methodology | Understanding approach |
| **QUESTION_BANK_SUMMARY.md** | Question coverage | Reviewing test scope |

---

## 🛠️ Common Commands

### Testing

```bash
# Pilot test (10 questions, ~2 min)
python test_harness.py --pilot

# Full test (50 questions, ~20 min)
python test_harness.py

# Test only ClimateGPT (no Llama needed)
python test_harness.py --climategpt-only

# Test specific questions
python test_harness.py --questions 1,2,3,4,5

# Verbose output
python test_harness.py --pilot --verbose
```

### Analysis

```bash
# Basic analysis
python analyze_results.py

# With visualizations
python analyze_results.py --visualize

# Export report
python analyze_results.py --report

# All at once
python analyze_results.py --visualize --report
```

### Verification

```bash
# Check setup
python verify_setup.py

# Test LM Studio connection
curl http://localhost:1234/v1/models
```

---

## ⚙️ Configuration

Edit `test_config.json`:

```json
{
  "climategpt": {
    "url": "http://localhost:8010"
  },
  "llama": {
    "url": "http://localhost:1234",
    "model": "meta-llama-3.1-8b-instruct@q4_k_m"
  }
}
```

**Note**: Model ID must match what's in LM Studio (check with `curl http://localhost:1234/v1/models`)

---

## 📈 Expected Results

**ClimateGPT**:
- ✅ High accuracy (real data from DuckDB)
- ✅ Specific numbers with units
- ✅ Source attribution
- Response time: 1000-2000ms average

**Meta Llama**:
- ❌ Low accuracy (no real data access)
- ❌ Generic or hallucinated responses
- ✅ Natural conversational tone
- Response time: 500-1000ms average

---

## 🐛 Troubleshooting

### ClimateGPT Not Running

```bash
# From project root
cd ..
make serve
```

### LM Studio Not Responding

1. Open LM Studio app
2. Go to "Local Server" tab
3. Verify model is loaded
4. Click "Start Server"
5. Test: `curl http://localhost:1234/v1/models`

### Import Errors

```bash
pip install -r requirements_testing.txt
```

### Wrong Model ID

Check actual model ID:
```bash
curl http://localhost:1234/v1/models | jq '.data[0].id'
```

Update in `test_config.json`:
```json
{
  "llama": {
    "model": "your-actual-model-id-here"
  }
}
```

---

## 🗑️ Cleanup After Testing

Once testing is complete and you have your results:

```bash
# From project root
cd ..
rm -rf testing/

# Or keep results and docs, remove only scripts
cd testing
rm test_harness.py analyze_results.py verify_setup.py
```

---

## 📦 Files You Can Safely Delete

After testing, you can remove:

**Immediately** (if you don't need to re-run):
- `test_harness.py`
- `analyze_results.py`
- `verify_setup.py`
- `extract_ground_truth.py`
- `requirements_testing.txt`

**Keep for Reference**:
- `test_results/` - Your test results
- `test_question_bank.json` - Question bank
- Documentation files - For future reference

**Or Remove Everything**:
```bash
cd ..
rm -rf testing/
```

---

## 🎓 Next Steps After Testing

1. ✅ Review results in `test_results/`
2. ✅ Identify top 5 improvement areas
3. ✅ Implement improvements in ClimateGPT source
4. ✅ Re-test to verify improvements
5. ✅ Document findings

---

## 📞 Getting Help

1. Check `QUICKSTART.md` for quick issues
2. Review `TEST_HARNESS_USAGE.md` for detailed docs
3. Run `python verify_setup.py` to check configuration
4. Check error messages carefully
5. Verify both services are running

---

## ✅ Status

- ✅ All scripts ready to use
- ✅ Configuration updated for your LM Studio models
- ✅ Documentation complete
- ✅ Ready to test!

---

**Start here**: Read [QUICKSTART.md](QUICKSTART.md) for 5-minute setup
**Configuration**: [test_config.json](test_config.json)
**Run test**: `python test_harness.py --pilot`

**Created**: 2025-11-02
**Status**: Production-ready ✅
