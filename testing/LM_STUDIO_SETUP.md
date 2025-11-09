# LM Studio Setup Guide

## Step-by-Step: Download and Configure Meta Llama

### Step 1: Open LM Studio

Launch the LM Studio application you just installed.

---

### Step 2: Download a Model

#### Option A: Recommended Model (Meta Llama 3.1 8B)

1. Click on the **"Search"** or **"Discover"** tab (🔍 icon on the left sidebar)

2. In the search bar, type: **`llama-3.1`**

3. You'll see several options. Look for:
   ```
   Meta-Llama-3.1-8B-Instruct
   ```

4. Click on the model card

5. You'll see different quantization options (file sizes):
   - **Q4_K_M** (~4.7 GB) - **RECOMMENDED** (good balance)
   - **Q5_K_M** (~5.7 GB) - Better quality
   - **Q8_0** (~8.5 GB) - Highest quality
   - **Q3_K_L** (~3.5 GB) - Faster, lower quality

6. Click **Download** on your preferred quantization
   - For most testing: **Q4_K_M** is perfect
   - If you have good hardware: **Q5_K_M** or **Q8_0**
   - If limited RAM/disk: **Q3_K_L**

7. Wait for download to complete (shows progress bar)

#### Option B: Smaller/Faster Models

If you have limited resources or want faster responses:

**Llama 3.2 3B Instruct** (~2 GB)
```
Search: llama-3.2
Model: Meta-Llama-3.2-3B-Instruct
Quantization: Q4_K_M (~2.0 GB)
```

**Llama 3.2 1B Instruct** (~1 GB)
```
Search: llama-3.2
Model: Meta-Llama-3.2-1B-Instruct
Quantization: Q4_K_M (~900 MB)
```

---

### Step 3: Load the Model

1. After download completes, go to **"Local Server"** tab (server icon on left)

2. At the top, you'll see **"Select a model to load"**

3. Click the dropdown and select your downloaded model:
   ```
   Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
   ```

4. Click **"Load Model"** (or it may load automatically)

5. Wait for model to load into memory (shows loading progress)

---

### Step 4: Start the Server

1. Once model is loaded, you'll see **"Start Server"** button

2. Click **"Start Server"**

3. Server should start on **http://localhost:1234**

4. You'll see:
   ```
   ✓ Server running on http://localhost:1234
   ```

---

### Step 5: Verify Server is Running

#### Option A: Use Browser
Open: http://localhost:1234/v1/models

You should see JSON response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama-3.1-8b-instruct",
      "object": "model",
      ...
    }
  ]
}
```

#### Option B: Use Terminal
```bash
curl http://localhost:1234/v1/models
```

---

### Step 6: Configure Test Harness

#### Find Your Model ID

In LM Studio, the model ID is shown in the server tab. It usually looks like:
- `meta-llama-3.1-8b-instruct`
- `meta-llama-3.2-3b-instruct`
- Or the full filename without `.gguf`

#### Update test_config.json

Edit the config file:

```bash
vim test_config.json
# or
nano test_config.json
# or open in any text editor
```

Update the `llama.model` field:

```json
{
  "llama": {
    "url": "http://localhost:1234",
    "endpoint": "/v1/chat/completions",
    "model": "meta-llama-3.1-8b-instruct",  ← Update this line
    "timeout": 30,
    "temperature": 0.1,
    "max_tokens": 500,
    "system_prompt": "You are an expert on climate emissions data..."
  }
}
```

**Important**: The model name should match what you see in LM Studio's server tab.

---

### Step 7: Test the Configuration

Run this quick test:

```bash
# Test that LM Studio is responding
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama-3.1-8b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
```

You should get a JSON response with the model's answer.

---

## Model Recommendations

### For Testing ClimateGPT

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **Llama 3.1 8B Q4_K_M** | 4.7GB | Medium | Good | **Recommended - Best balance** |
| Llama 3.1 8B Q5_K_M | 5.7GB | Slower | Better | Higher quality responses |
| Llama 3.2 3B Q4_K_M | 2.0GB | Fast | Fair | Limited resources |
| Llama 3.2 1B Q4_K_M | 0.9GB | Very Fast | Basic | Very limited resources |

### Hardware Requirements

| Model | Min RAM | Recommended RAM | Notes |
|-------|---------|-----------------|-------|
| Llama 3.1 8B | 8 GB | 16 GB | May be slow on 8GB |
| Llama 3.2 3B | 4 GB | 8 GB | Good for laptops |
| Llama 3.2 1B | 2 GB | 4 GB | Lightweight |

---

## Troubleshooting

### Model Won't Load

**Error**: "Not enough memory"

**Solution**:
1. Close other applications
2. Try a smaller quantization (Q3_K_L instead of Q4_K_M)
3. Try Llama 3.2 3B instead of 3.1 8B

### Server Won't Start

**Error**: Port already in use

**Solution**:
```bash
# Check if something is using port 1234
lsof -i :1234

# Kill the process if needed
kill -9 <PID>

# Or change the port in LM Studio settings
```

### Model ID Doesn't Match

**Error**: "Model not found" when running tests

**Solution**:
1. Check the exact model ID in LM Studio's "Local Server" tab
2. Copy it exactly into `test_config.json`
3. Common IDs:
   - `meta-llama-3.1-8b-instruct`
   - `llama-3.1-8b-instruct`
   - Or the full GGUF filename without extension

### Slow Responses

**Issue**: Taking 10+ seconds per response

**Solutions**:
1. Switch to smaller model (3B or 1B)
2. Use lower quantization (Q3_K_L)
3. Close other apps to free RAM
4. Enable GPU acceleration in LM Studio settings:
   - Go to Settings (⚙️)
   - Enable "Use GPU if available"

---

## Quick Reference

### Common Model IDs for test_config.json

```json
// Llama 3.1 8B (Recommended)
"model": "meta-llama-3.1-8b-instruct"

// Llama 3.2 3B (Faster)
"model": "meta-llama-3.2-3b-instruct"

// Llama 3.2 1B (Lightweight)
"model": "meta-llama-3.2-1b-instruct"
```

### LM Studio Default Settings

```
URL: http://localhost:1234
Endpoint: /v1/chat/completions
API: OpenAI-compatible
```

---

## Testing Your Setup

### Quick Test Script

Create a file `test_llm_studio.py`:

```python
import requests

# Test LM Studio
response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "model": "meta-llama-3.1-8b-instruct",  # Your model ID
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ]
    }
)

print("Status:", response.status_code)
print("Response:", response.json()['choices'][0]['message']['content'])
```

Run it:
```bash
python test_llm_studio.py
```

Expected output:
```
Status: 200
Response: 2 + 2 equals 4.
```

---

## Next Steps

Once LM Studio is running:

1. ✅ Model downloaded and loaded
2. ✅ Server running on http://localhost:1234
3. ✅ Model ID configured in test_config.json
4. ✅ Test script confirms it works

**You're ready to run the test harness!**

```bash
# Run pilot test
python test_harness.py --pilot

# Analyze results
python analyze_results.py
```

---

## Alternative: Using Different Models

### If You Want to Test Other LLMs

You can also download and test:

**Mistral Models** (good alternative):
```
Search: mistral
Model: Mistral-7B-Instruct
Quantization: Q4_K_M
```

**Qwen Models** (good for factual tasks):
```
Search: qwen
Model: Qwen2.5-7B-Instruct
Quantization: Q4_K_M
```

**Note**: Update `test_config.json` with the correct model ID for each.

---

## FAQ

**Q: Which quantization should I use?**
A: Q4_K_M for most cases. It's the best balance of quality and size.

**Q: Can I use multiple models?**
A: Yes! Create different config files:
```bash
# Test with Llama 3.1
python test_harness.py --config config_llama31.json

# Test with Llama 3.2
python test_harness.py --config config_llama32.json
```

**Q: How do I know if my GPU is being used?**
A: Check LM Studio settings and Task Manager/Activity Monitor. GPU usage should be visible.

**Q: Can I run LM Studio on a different port?**
A: Yes, in LM Studio settings, change the port. Then update `test_config.json`:
```json
{
  "llama": {
    "url": "http://localhost:5000"  // Your new port
  }
}
```

**Q: Do I need to keep LM Studio UI open?**
A: Yes, while testing. The UI manages the server and model.

---

**Last Updated**: 2025-11-02
**Recommended**: Meta-Llama-3.1-8B-Instruct Q4_K_M
**Status**: Ready to use ✅
