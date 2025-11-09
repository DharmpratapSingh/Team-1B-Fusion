#!/usr/bin/env python3
"""
Automated Test Harness for Comparative LLM Testing

Tests ClimateGPT against other LLMs (Meta Llama via LM Studio) using
a comprehensive question bank.

Usage:
    # Test both systems
    python test_harness.py

    # Test only ClimateGPT
    python test_harness.py --climategpt-only

    # Test only Llama
    python test_harness.py --llama-only

    # Test specific questions
    python test_harness.py --questions 1,2,3,4,5

    # Run pilot test (first 10 questions)
    python test_harness.py --pilot

    # Custom config
    python test_harness.py --config my_config.json
"""

import json
import requests
import time
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from a single test."""
    question_id: int
    question: str
    category: str
    sector: str
    level: str
    grain: str
    difficulty: str
    system: str  # 'climategpt' or 'llama'
    response: Optional[str]
    response_time_ms: float
    status_code: Optional[int]
    error: Optional[str]
    timestamp: str


class TestConfig:
    """Test configuration."""

    def __init__(self, config_path: str = "test_config.json"):
        """Load configuration from file."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = {}

        # ClimateGPT settings
        self.climategpt_url = config.get('climategpt', {}).get('url', 'http://localhost:8010')
        self.climategpt_endpoint = config.get('climategpt', {}).get('endpoint', '/query')
        self.climategpt_timeout = config.get('climategpt', {}).get('timeout', 30)

        # LM Studio settings
        self.llama_url = config.get('llama', {}).get('url', 'http://localhost:1234')
        self.llama_endpoint = config.get('llama', {}).get('endpoint', '/v1/chat/completions')
        self.llama_model = config.get('llama', {}).get('model', 'meta-llama-3.1-8b-instruct')
        self.llama_timeout = config.get('llama', {}).get('timeout', 30)
        self.llama_temperature = config.get('llama', {}).get('temperature', 0.1)
        self.llama_max_tokens = config.get('llama', {}).get('max_tokens', 500)
        self.llama_system_prompt = config.get('llama', {}).get(
            'system_prompt',
            'You are an expert on climate emissions data. Provide accurate, concise answers with specific numbers when possible. Use metric tons of CO2 (MtCO2) as the unit.'
        )

        # Test settings
        self.question_bank_path = config.get('test', {}).get('question_bank', 'test_question_bank.json')
        self.output_dir = config.get('test', {}).get('output_dir', 'test_results')
        self.delay_between_requests = config.get('test', {}).get('delay_between_requests', 1.0)
        self.max_retries = config.get('test', {}).get('max_retries', 2)
        self.retry_delay = config.get('test', {}).get('retry_delay', 2.0)


class TestHarness:
    """Main test harness for comparative LLM testing."""

    def __init__(self, config: TestConfig):
        """Initialize test harness."""
        self.config = config
        self.results: List[TestResult] = []
        self.questions: List[Dict[str, Any]] = []

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_questions(self, question_ids: Optional[List[int]] = None) -> None:
        """Load questions from question bank."""
        logger.info(f"Loading questions from {self.config.question_bank_path}")

        with open(self.config.question_bank_path, 'r') as f:
            data = json.load(f)
            all_questions = data['questions']

        if question_ids:
            # Filter to specific questions
            self.questions = [q for q in all_questions if q['id'] in question_ids]
            logger.info(f"Loaded {len(self.questions)} specific questions")
        else:
            self.questions = all_questions
            logger.info(f"Loaded {len(self.questions)} questions")

    def test_climategpt(self, question_text: str, question_id: int) -> TestResult:
        """Test ClimateGPT with a question."""
        q_meta = next(q for q in self.questions if q['id'] == question_id)

        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.time()
            try:
                url = f"{self.config.climategpt_url}{self.config.climategpt_endpoint}"
                logger.debug(f"Calling ClimateGPT: {url}")

                response = requests.post(
                    url,
                    json={"question": question_text},
                    timeout=self.config.climategpt_timeout
                )
                elapsed = (time.time() - start_time) * 1000

                # Extract response text
                response_data = response.json()
                if isinstance(response_data, dict):
                    # Try common response fields
                    response_text = (
                        response_data.get('answer') or
                        response_data.get('response') or
                        response_data.get('result') or
                        json.dumps(response_data, indent=2)
                    )
                else:
                    response_text = str(response_data)

                return TestResult(
                    question_id=question_id,
                    question=question_text,
                    category=q_meta['category'],
                    sector=q_meta['sector'],
                    level=q_meta['level'],
                    grain=q_meta['grain'],
                    difficulty=q_meta['difficulty'],
                    system='climategpt',
                    response=response_text,
                    response_time_ms=round(elapsed, 2),
                    status_code=response.status_code,
                    error=None if response.status_code == 200 else f"HTTP {response.status_code}",
                    timestamp=datetime.now().isoformat()
                )

            except requests.exceptions.Timeout:
                elapsed = (time.time() - start_time) * 1000
                error_msg = f"Timeout after {self.config.climategpt_timeout}s (attempt {attempt}/{self.config.max_retries})"
                logger.warning(f"Q{question_id} ClimateGPT: {error_msg}")

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return TestResult(
                        question_id=question_id,
                        question=question_text,
                        category=q_meta['category'],
                        sector=q_meta['sector'],
                        level=q_meta['level'],
                        grain=q_meta['grain'],
                        difficulty=q_meta['difficulty'],
                        system='climategpt',
                        response=None,
                        response_time_ms=round(elapsed, 2),
                        status_code=None,
                        error=error_msg,
                        timestamp=datetime.now().isoformat()
                    )

            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Q{question_id} ClimateGPT error: {error_msg}")

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return TestResult(
                        question_id=question_id,
                        question=question_text,
                        category=q_meta['category'],
                        sector=q_meta['sector'],
                        level=q_meta['level'],
                        grain=q_meta['grain'],
                        difficulty=q_meta['difficulty'],
                        system='climategpt',
                        response=None,
                        response_time_ms=round(elapsed, 2),
                        status_code=None,
                        error=error_msg,
                        timestamp=datetime.now().isoformat()
                    )

    def test_llama(self, question_text: str, question_id: int) -> TestResult:
        """Test Llama via LM Studio with a question."""
        q_meta = next(q for q in self.questions if q['id'] == question_id)

        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.time()
            try:
                url = f"{self.config.llama_url}{self.config.llama_endpoint}"
                logger.debug(f"Calling Llama: {url}")

                response = requests.post(
                    url,
                    json={
                        "model": self.config.llama_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": self.config.llama_system_prompt
                            },
                            {
                                "role": "user",
                                "content": question_text
                            }
                        ],
                        "temperature": self.config.llama_temperature,
                        "max_tokens": self.config.llama_max_tokens
                    },
                    timeout=self.config.llama_timeout
                )
                elapsed = (time.time() - start_time) * 1000

                response_data = response.json()
                response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

                return TestResult(
                    question_id=question_id,
                    question=question_text,
                    category=q_meta['category'],
                    sector=q_meta['sector'],
                    level=q_meta['level'],
                    grain=q_meta['grain'],
                    difficulty=q_meta['difficulty'],
                    system='llama',
                    response=response_text,
                    response_time_ms=round(elapsed, 2),
                    status_code=response.status_code,
                    error=None if response.status_code == 200 else f"HTTP {response.status_code}",
                    timestamp=datetime.now().isoformat()
                )

            except requests.exceptions.Timeout:
                elapsed = (time.time() - start_time) * 1000
                error_msg = f"Timeout after {self.config.llama_timeout}s (attempt {attempt}/{self.config.max_retries})"
                logger.warning(f"Q{question_id} Llama: {error_msg}")

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return TestResult(
                        question_id=question_id,
                        question=question_text,
                        category=q_meta['category'],
                        sector=q_meta['sector'],
                        level=q_meta['level'],
                        grain=q_meta['grain'],
                        difficulty=q_meta['difficulty'],
                        system='llama',
                        response=None,
                        response_time_ms=round(elapsed, 2),
                        status_code=None,
                        error=error_msg,
                        timestamp=datetime.now().isoformat()
                    )

            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Q{question_id} Llama error: {error_msg}")

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return TestResult(
                        question_id=question_id,
                        question=question_text,
                        category=q_meta['category'],
                        sector=q_meta['sector'],
                        level=q_meta['level'],
                        grain=q_meta['grain'],
                        difficulty=q_meta['difficulty'],
                        system='llama',
                        response=None,
                        response_time_ms=round(elapsed, 2),
                        status_code=None,
                        error=error_msg,
                        timestamp=datetime.now().isoformat()
                    )

    def check_services(self, test_climategpt: bool, test_llama: bool) -> bool:
        """Check if required services are running."""
        all_ok = True

        if test_climategpt:
            try:
                url = f"{self.config.climategpt_url}/health"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ ClimateGPT is running at {self.config.climategpt_url}")
                else:
                    logger.error(f"✗ ClimateGPT health check failed: HTTP {response.status_code}")
                    all_ok = False
            except Exception as e:
                logger.error(f"✗ ClimateGPT is not reachable at {self.config.climategpt_url}")
                logger.error(f"  Error: {e}")
                logger.error(f"  Please start ClimateGPT: make serve")
                all_ok = False

        if test_llama:
            try:
                url = f"{self.config.llama_url}/v1/models"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ LM Studio is running at {self.config.llama_url}")
                else:
                    logger.error(f"✗ LM Studio health check failed: HTTP {response.status_code}")
                    all_ok = False
            except Exception as e:
                logger.error(f"✗ LM Studio is not reachable at {self.config.llama_url}")
                logger.error(f"  Error: {e}")
                logger.error(f"  Please start LM Studio server")
                all_ok = False

        return all_ok

    def run_tests(self, test_climategpt: bool = True, test_llama: bool = True) -> None:
        """Run all tests."""
        if not self.questions:
            logger.error("No questions loaded. Call load_questions() first.")
            return

        # Check services
        logger.info("\nChecking services...")
        if not self.check_services(test_climategpt, test_llama):
            logger.error("\nService check failed. Please fix errors above and try again.")
            sys.exit(1)

        total = len(self.questions)
        systems = []
        if test_climategpt:
            systems.append('climategpt')
        if test_llama:
            systems.append('llama')

        logger.info(f"\nStarting tests: {total} questions × {len(systems)} systems = {total * len(systems)} tests")
        logger.info(f"Systems: {', '.join(systems)}")
        logger.info(f"Delay between requests: {self.config.delay_between_requests}s")
        logger.info("-" * 80)

        start_time = time.time()

        for i, question in enumerate(self.questions, 1):
            q_id = question['id']
            q_text = question['question']

            logger.info(f"\n[{i}/{total}] Q{q_id}: {q_text[:60]}{'...' if len(q_text) > 60 else ''}")

            # Test ClimateGPT
            if test_climategpt:
                logger.info(f"  → Testing ClimateGPT...")
                result = self.test_climategpt(q_text, q_id)
                self.results.append(result)

                if result.error:
                    logger.error(f"    ✗ Error: {result.error}")
                else:
                    logger.info(f"    ✓ Response: {len(result.response) if result.response else 0} chars, {result.response_time_ms}ms")

                time.sleep(self.config.delay_between_requests)

            # Test Llama
            if test_llama:
                logger.info(f"  → Testing Llama...")
                result = self.test_llama(q_text, q_id)
                self.results.append(result)

                if result.error:
                    logger.error(f"    ✗ Error: {result.error}")
                else:
                    logger.info(f"    ✓ Response: {len(result.response) if result.response else 0} chars, {result.response_time_ms}ms")

                time.sleep(self.config.delay_between_requests)

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing complete! Total time: {elapsed:.1f}s")
        logger.info(f"Total tests run: {len(self.results)}")

        # Summary
        errors = sum(1 for r in self.results if r.error)
        success = len(self.results) - errors
        logger.info(f"Success: {success}, Errors: {errors}")

    def save_results(self) -> str:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_results_{timestamp}.json"
        filepath = Path(self.config.output_dir) / filename

        # Organize results by question
        results_by_question = {}
        for result in self.results:
            q_id = result.question_id
            if q_id not in results_by_question:
                results_by_question[q_id] = {
                    'question_id': q_id,
                    'question': result.question,
                    'category': result.category,
                    'sector': result.sector,
                    'level': result.level,
                    'grain': result.grain,
                    'difficulty': result.difficulty,
                    'results': {}
                }

            results_by_question[q_id]['results'][result.system] = {
                'response': result.response,
                'response_time_ms': result.response_time_ms,
                'status_code': result.status_code,
                'error': result.error,
                'timestamp': result.timestamp
            }

        output = {
            'metadata': {
                'test_date': datetime.now().isoformat(),
                'total_questions': len(self.questions),
                'total_tests': len(self.results),
                'config': {
                    'climategpt_url': self.config.climategpt_url,
                    'llama_url': self.config.llama_url,
                    'llama_model': self.config.llama_model
                }
            },
            'results': list(results_by_question.values())
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {filepath}")
        return str(filepath)

    def save_csv(self) -> str:
        """Save results to CSV file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_results_{timestamp}.csv"
        filepath = Path(self.config.output_dir) / filename

        import csv

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'question_id', 'question', 'category', 'sector', 'level', 'grain', 'difficulty',
                'system', 'response', 'response_time_ms', 'status_code', 'error', 'timestamp'
            ])

            # Data
            for result in self.results:
                writer.writerow([
                    result.question_id,
                    result.question,
                    result.category,
                    result.sector,
                    result.level,
                    result.grain,
                    result.difficulty,
                    result.system,
                    result.response or '',
                    result.response_time_ms,
                    result.status_code or '',
                    result.error or '',
                    result.timestamp
                ])

        logger.info(f"CSV results saved to: {filepath}")
        return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated test harness for comparative LLM testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test both systems
  python test_harness.py

  # Test only ClimateGPT
  python test_harness.py --climategpt-only

  # Test only Llama
  python test_harness.py --llama-only

  # Run pilot test (first 10 questions)
  python test_harness.py --pilot

  # Test specific questions
  python test_harness.py --questions 1,2,3,4,5

  # Use custom config
  python test_harness.py --config my_config.json
        """
    )

    parser.add_argument(
        '--config',
        default='test_config.json',
        help='Path to configuration file (default: test_config.json)'
    )
    parser.add_argument(
        '--climategpt-only',
        action='store_true',
        help='Test only ClimateGPT'
    )
    parser.add_argument(
        '--llama-only',
        action='store_true',
        help='Test only Llama'
    )
    parser.add_argument(
        '--pilot',
        action='store_true',
        help='Run pilot test (first 10 questions)'
    )
    parser.add_argument(
        '--questions',
        type=str,
        help='Comma-separated list of question IDs to test (e.g., 1,2,3,4,5)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load config
    config = TestConfig(args.config)

    # Create harness
    harness = TestHarness(config)

    # Determine which questions to test
    if args.questions:
        question_ids = [int(q.strip()) for q in args.questions.split(',')]
        harness.load_questions(question_ids)
    elif args.pilot:
        # Pilot questions: 1, 7, 13, 4, 5, 11, 17, 23, 48, 50
        pilot_ids = [1, 7, 13, 4, 5, 11, 17, 23, 48, 50]
        harness.load_questions(pilot_ids)
        logger.info("Running pilot test with 10 representative questions")
    else:
        harness.load_questions()

    # Determine which systems to test
    test_climategpt = not args.llama_only
    test_llama = not args.climategpt_only

    # Run tests
    harness.run_tests(test_climategpt=test_climategpt, test_llama=test_llama)

    # Save results
    harness.save_results()
    harness.save_csv()

    logger.info("\n✓ Test harness complete!")


if __name__ == "__main__":
    main()
