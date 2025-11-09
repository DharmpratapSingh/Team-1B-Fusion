#!/usr/bin/env python3
"""
Results Analysis Script for Comparative LLM Testing

Analyzes test results from test_harness.py and generates:
- Summary statistics
- Performance comparisons
- Error analysis
- Visualizations (optional)

Usage:
    # Analyze latest results
    python analyze_results.py

    # Analyze specific results file
    python analyze_results.py --file test_results/test_results_20251102_143022.json

    # Generate visualization charts
    python analyze_results.py --visualize

    # Export summary report
    python analyze_results.py --report
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed. Install with: pip install pandas")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


class ResultsAnalyzer:
    """Analyze test results."""

    def __init__(self, results_file: str):
        """Initialize analyzer."""
        self.results_file = results_file
        self.data = None
        self.df = None

    def load_results(self) -> bool:
        """Load results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded results from: {self.results_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False

    def to_dataframe(self) -> bool:
        """Convert results to pandas DataFrame."""
        if not HAS_PANDAS:
            logger.warning("pandas not available, skipping DataFrame conversion")
            return False

        rows = []
        for question in self.data['results']:
            q_id = question['question_id']
            q_text = question['question']
            category = question['category']
            sector = question['sector']
            level = question['level']
            grain = question['grain']
            difficulty = question['difficulty']

            for system, result in question['results'].items():
                rows.append({
                    'question_id': q_id,
                    'question': q_text,
                    'category': category,
                    'sector': sector,
                    'level': level,
                    'grain': grain,
                    'difficulty': difficulty,
                    'system': system,
                    'response': result['response'],
                    'response_time_ms': result['response_time_ms'],
                    'status_code': result['status_code'],
                    'error': result['error'],
                    'has_error': result['error'] is not None,
                    'response_length': len(result['response']) if result['response'] else 0
                })

        self.df = pd.DataFrame(rows)
        logger.info(f"Created DataFrame with {len(self.df)} rows")
        return True

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        metadata = self.data['metadata']
        print(f"\nTest Date: {metadata['test_date']}")
        print(f"Total Questions: {metadata['total_questions']}")
        print(f"Total Tests: {metadata['total_tests']}")

        print(f"\nConfiguration:")
        print(f"  ClimateGPT: {metadata['config']['climategpt_url']}")
        print(f"  Llama: {metadata['config']['llama_url']} ({metadata['config']['llama_model']})")

        # Overall statistics
        print("\n" + "-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)

        systems = set()
        total_tests = 0
        errors_by_system = {}
        response_times = {}

        for question in self.data['results']:
            for system, result in question['results'].items():
                systems.add(system)
                total_tests += 1

                if system not in errors_by_system:
                    errors_by_system[system] = 0
                    response_times[system] = []

                if result['error']:
                    errors_by_system[system] += 1

                if result['response_time_ms']:
                    response_times[system].append(result['response_time_ms'])

        for system in sorted(systems):
            tests_count = len(response_times[system]) + errors_by_system[system]
            success_count = tests_count - errors_by_system[system]
            success_rate = (success_count / tests_count * 100) if tests_count > 0 else 0

            print(f"\n{system.upper()}:")
            print(f"  Total tests: {tests_count}")
            print(f"  Successful: {success_count} ({success_rate:.1f}%)")
            print(f"  Errors: {errors_by_system[system]}")

            if response_times[system]:
                times = response_times[system]
                print(f"  Response time:")
                print(f"    Mean: {sum(times) / len(times):.1f}ms")
                print(f"    Median: {sorted(times)[len(times) // 2]:.1f}ms")
                print(f"    Min: {min(times):.1f}ms")
                print(f"    Max: {max(times):.1f}ms")

    def print_comparison(self):
        """Print detailed comparison between systems."""
        if not HAS_PANDAS or self.df is None:
            logger.warning("DataFrame not available, skipping comparison")
            return

        print("\n" + "-" * 80)
        print("PERFORMANCE BY CATEGORY")
        print("-" * 80)

        for category in self.df['category'].unique():
            print(f"\n{category.upper()}:")
            cat_df = self.df[self.df['category'] == category]

            for system in self.df['system'].unique():
                sys_df = cat_df[cat_df['system'] == system]
                success = len(sys_df[~sys_df['has_error']])
                total = len(sys_df)
                avg_time = sys_df['response_time_ms'].mean()

                print(f"  {system}: {success}/{total} success, {avg_time:.1f}ms avg")

        print("\n" + "-" * 80)
        print("PERFORMANCE BY SECTOR")
        print("-" * 80)

        for sector in sorted(self.df['sector'].unique()):
            print(f"\n{sector}:")
            sector_df = self.df[self.df['sector'] == sector]

            for system in self.df['system'].unique():
                sys_df = sector_df[sector_df['system'] == system]
                success = len(sys_df[~sys_df['has_error']])
                total = len(sys_df)
                avg_time = sys_df['response_time_ms'].mean()

                print(f"  {system}: {success}/{total} success, {avg_time:.1f}ms avg")

        print("\n" + "-" * 80)
        print("PERFORMANCE BY DIFFICULTY")
        print("-" * 80)

        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in self.df['difficulty'].unique():
                print(f"\n{difficulty.upper()}:")
                diff_df = self.df[self.df['difficulty'] == difficulty]

                for system in self.df['system'].unique():
                    sys_df = diff_df[diff_df['system'] == system]
                    success = len(sys_df[~sys_df['has_error']])
                    total = len(sys_df)
                    avg_time = sys_df['response_time_ms'].mean()

                    print(f"  {system}: {success}/{total} success, {avg_time:.1f}ms avg")

    def print_errors(self):
        """Print error analysis."""
        print("\n" + "-" * 80)
        print("ERROR ANALYSIS")
        print("-" * 80)

        errors = []
        for question in self.data['results']:
            for system, result in question['results'].items():
                if result['error']:
                    errors.append({
                        'question_id': question['question_id'],
                        'question': question['question'][:60] + '...' if len(question['question']) > 60 else question['question'],
                        'system': system,
                        'error': result['error']
                    })

        if not errors:
            print("\n✓ No errors detected!")
        else:
            print(f"\nTotal errors: {len(errors)}\n")
            for err in errors:
                print(f"Q{err['question_id']} ({err['system']}): {err['error']}")
                print(f"  Question: {err['question']}")
                print()

    def print_sample_responses(self, n: int = 3):
        """Print sample responses from both systems."""
        print("\n" + "-" * 80)
        print(f"SAMPLE RESPONSES (first {n} questions)")
        print("-" * 80)

        for question in self.data['results'][:n]:
            print(f"\nQ{question['question_id']}: {question['question']}")
            print(f"Category: {question['category']}, Sector: {question['sector']}, "
                  f"Level: {question['level']}, Difficulty: {question['difficulty']}")
            print()

            for system, result in question['results'].items():
                print(f"  {system.upper()}:")
                if result['error']:
                    print(f"    Error: {result['error']}")
                else:
                    response = result['response'][:200] + '...' if len(result['response']) > 200 else result['response']
                    print(f"    Response: {response}")
                    print(f"    Time: {result['response_time_ms']}ms")
                print()

    def generate_visualizations(self, output_dir: str = "test_results"):
        """Generate visualization charts."""
        if not HAS_VISUALIZATION:
            logger.warning("Visualization libraries not available. Install with: pip install matplotlib seaborn")
            return

        if not HAS_PANDAS or self.df is None:
            logger.warning("DataFrame not available, skipping visualizations")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Response time comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='system', y='response_time_ms')
        plt.title('Response Time Comparison')
        plt.ylabel('Response Time (ms)')
        plt.xlabel('System')
        plt.savefig(f"{output_dir}/response_time_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_dir}/response_time_comparison.png")
        plt.close()

        # 2. Success rate by category
        fig, ax = plt.subplots(figsize=(10, 6))
        success_by_cat = self.df.groupby(['category', 'system']).apply(
            lambda x: (~x['has_error']).sum() / len(x) * 100
        ).reset_index(name='success_rate')
        sns.barplot(data=success_by_cat, x='category', y='success_rate', hue='system', ax=ax)
        plt.title('Success Rate by Category')
        plt.ylabel('Success Rate (%)')
        plt.xlabel('Category')
        plt.ylim(0, 105)
        plt.legend(title='System')
        plt.savefig(f"{output_dir}/success_rate_by_category.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_dir}/success_rate_by_category.png")
        plt.close()

        # 3. Response time by sector
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='sector', y='response_time_ms', hue='system')
        plt.title('Response Time by Sector')
        plt.ylabel('Response Time (ms)')
        plt.xlabel('Sector')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='System')
        plt.savefig(f"{output_dir}/response_time_by_sector.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_dir}/response_time_by_sector.png")
        plt.close()

        # 4. Response length comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='system', y='response_length')
        plt.title('Response Length Comparison')
        plt.ylabel('Response Length (characters)')
        plt.xlabel('System')
        plt.savefig(f"{output_dir}/response_length_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_dir}/response_length_comparison.png")
        plt.close()

        logger.info(f"\n✓ Visualizations saved to {output_dir}/")

    def export_report(self, output_file: str = None):
        """Export summary report to text file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"test_results/analysis_report_{timestamp}.txt"

        # Redirect stdout to file
        import io
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        # Print all analysis
        self.print_summary()
        self.print_comparison()
        self.print_errors()
        self.print_sample_responses(5)

        # Restore stdout
        sys.stdout = old_stdout

        # Write to file
        with open(output_file, 'w') as f:
            f.write(buffer.getvalue())

        logger.info(f"\n✓ Report saved to: {output_file}")


def find_latest_results(results_dir: str = "test_results") -> str:
    """Find the most recent results file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return None

    json_files = list(results_path.glob("test_results_*.json"))
    if not json_files:
        logger.error(f"No result files found in {results_dir}")
        return None

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze comparative LLM test results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--file',
        help='Path to results JSON file (default: latest in test_results/)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization charts'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Export summary report to file'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary to console'
    )

    args = parser.parse_args()

    # Find results file
    if args.file:
        results_file = args.file
    else:
        results_file = find_latest_results()

    if not results_file:
        logger.error("No results file specified or found")
        sys.exit(1)

    # Analyze
    analyzer = ResultsAnalyzer(results_file)

    if not analyzer.load_results():
        sys.exit(1)

    analyzer.to_dataframe()

    # Print to console
    if not args.no_summary:
        analyzer.print_summary()
        analyzer.print_comparison()
        analyzer.print_errors()
        analyzer.print_sample_responses(3)

    # Generate visualizations
    if args.visualize:
        analyzer.generate_visualizations()

    # Export report
    if args.report:
        analyzer.export_report()

    logger.info("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
