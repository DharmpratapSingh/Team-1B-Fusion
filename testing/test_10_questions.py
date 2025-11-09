#!/usr/bin/env python3
"""
Quick test script for 10 questions using run_llm.py
Tests both ClimateGPT (default) and Meta Llama (via LM Studio)
"""

import json
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Load questions
QUESTIONS_FILE = Path(__file__).parent / "test_question_bank.json"
OUTPUT_DIR = Path(__file__).parent / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_questions(n: int = 10) -> List[Dict[str, Any]]:
    """Load first N questions from question bank."""
    with open(QUESTIONS_FILE, 'r') as f:
        data = json.load(f)
        return data['questions'][:n]

def run_llm_query(question: str, llm_type: str = "climategpt") -> Dict[str, Any]:
    """
    Run a query using run_llm.py
    
    Args:
        question: The question to ask
        llm_type: "climategpt" or "llama"
    
    Returns:
        Dict with response, response_time_ms, error, etc.
    """
    start_time = time.time()
    
    # Prepare environment
    env = os.environ.copy()
    
    if llm_type == "llama":
        # Use LM Studio
        env["OPENAI_BASE_URL"] = "http://localhost:1234/v1"
        env["MODEL"] = "meta-llama-3.1-8b-instruct@q5_k_m"
        env.pop("OPENAI_API_KEY", None)  # Remove auth for LM Studio
    else:
        # Use default ClimateGPT (erasmus.ai)
        # Keep default env vars
        pass
    
    try:
        # Run run_llm.py
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "run_llm.py"),
                question
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            cwd=str(Path(__file__).parent.parent)
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Extract answer (look for "=== ANSWER ===" section)
        output = result.stdout
        if "=== ANSWER ===" in output:
            answer = output.split("=== ANSWER ===")[1].strip()
        else:
            answer = output[-2000:] if len(output) > 2000 else output  # Last 2000 chars
        
        return {
            "response": answer,
            "response_time_ms": round(elapsed_ms, 2),
            "status": "success",
            "error": None,
            "stdout": output,
            "stderr": result.stderr
        }
    
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "response": None,
            "response_time_ms": round(elapsed_ms, 2),
            "status": "timeout",
            "error": "Timeout after 120 seconds",
            "stdout": "",
            "stderr": ""
        }
    
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "response": None,
            "response_time_ms": round(elapsed_ms, 2),
            "status": "error",
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }

def main():
    print("=" * 60)
    print("Testing 10 Questions on Both LLMs")
    print("=" * 60)
    print()
    
    # Load first 10 questions
    questions = load_questions(10)
    print(f"Loaded {len(questions)} questions\n")
    
    # Wait for MCP server to be ready
    print("Waiting for MCP server to be ready...")
    max_wait = 30
    for i in range(max_wait):
        try:
            resp = requests.get("http://localhost:8010/list_files", timeout=2)
            if resp.status_code == 200:
                print("✅ MCP server is ready\n")
                break
        except:
            if i < max_wait - 1:
                time.sleep(1)
            else:
                print("⚠️  MCP server not responding, but continuing anyway...\n")
    
    # Test each question on both LLMs
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/10: {q['question']}")
        print(f"Category: {q['category']} | Sector: {q['sector']} | Level: {q['level']}")
        print(f"{'='*60}\n")
        
        # Test ClimateGPT
        print(f"[{i}/10] Testing ClimateGPT...")
        climategpt_result = run_llm_query(q['question'], "climategpt")
        print(f"  Time: {climategpt_result['response_time_ms']:.1f}ms")
        if climategpt_result['status'] == 'success':
            print(f"  ✅ Success")
            print(f"  Answer preview: {climategpt_result['response'][:100]}...")
        else:
            print(f"  ❌ {climategpt_result['status']}: {climategpt_result.get('error', 'Unknown error')}")
        
        time.sleep(1)  # Rate limiting
        
        # Test Llama
        print(f"[{i}/10] Testing Llama...")
        llama_result = run_llm_query(q['question'], "llama")
        print(f"  Time: {llama_result['response_time_ms']:.1f}ms")
        if llama_result['status'] == 'success':
            print(f"  ✅ Success")
            print(f"  Answer preview: {llama_result['response'][:100]}...")
        else:
            print(f"  ❌ {llama_result['status']}: {llama_result.get('error', 'Unknown error')}")
        
        # Store result
        results.append({
            "question_id": q['id'],
            "question": q['question'],
            "category": q['category'],
            "sector": q['sector'],
            "level": q['level'],
            "grain": q['grain'],
            "difficulty": q['difficulty'],
            "climategpt": climategpt_result,
            "llama": llama_result,
            "timestamp": datetime.now().isoformat()
        })
        
        time.sleep(1)  # Rate limiting between questions
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"test_10q_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "test_type": "10_question_pilot",
                "timestamp": timestamp,
                "total_questions": len(questions)
            },
            "results": results
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    climategpt_success = sum(1 for r in results if r['climategpt']['status'] == 'success')
    llama_success = sum(1 for r in results if r['llama']['status'] == 'success')
    
    climategpt_avg_time = sum(r['climategpt']['response_time_ms'] for r in results) / len(results)
    llama_avg_time = sum(r['llama']['response_time_ms'] for r in results) / len(results)
    
    print(f"\nClimateGPT:")
    print(f"  Success: {climategpt_success}/{len(questions)} ({100*climategpt_success/len(questions):.1f}%)")
    print(f"  Avg Time: {climategpt_avg_time:.1f}ms")
    
    print(f"\nLlama:")
    print(f"  Success: {llama_success}/{len(questions)} ({100*llama_success/len(questions):.1f}%)")
    print(f"  Avg Time: {llama_avg_time:.1f}ms")
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

