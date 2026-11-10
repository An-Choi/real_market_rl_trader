"""Check saved completion evidence and export a machine-readable artifact index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN = ROOT / 'runs/completion-v3'


def read(path):
    return json.loads(path.read_text())


def verify():
    python_log = WORKSPACE / '.runtime/logs/completion-v3-final-python-tests-unrestricted.log'
    python_result = re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s', python_log.read_text())
    if python_result is None or int(python_result[1]) != 622:
        raise ValueError('Final Python test run did not pass')
    java_log = WORKSPACE / '.runtime/logs/completion-v3-final-java-tests.log'
    if 'BUILD SUCCESSFUL' not in java_log.read_text():
        raise ValueError('Final Java build did not pass')
    xmls = sorted((WORKSPACE / 'rl-trading-core/build/test-results/test').glob('TEST-*.xml'))
    suites = [ET.parse(path).getroot().attrib for path in xmls]
    java = {key: sum(int(suite.get(key, 0)) for suite in suites)
            for key in ('tests', 'failures', 'errors', 'skipped')}
    if java != {'tests': 165, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError(f'Incomplete Java tests: {java}')
    completions = {}
    for folder in ('global-risk-study', 'global-risk-study/frozen_evaluation',
                   'predictive-diagnostics', 'global-policy-diagnostics', 'data-recovery',
                   'data-normalization'):
        completed = read(RUN / folder / 'completion.json')
        if not completed['completed']:
            raise ValueError(f'Incomplete: {folder}')
        completions[folder] = completed
    assets = read(RUN / 'result-assets/manifest.json')
    if assets['frozen_test_rows'] != 135 or assets['stress_rows'] != 495:
        raise ValueError('Incomplete result assets')
    for filename, checksum in assets['source_sha256'].items():
        if hashlib.sha256((RUN / filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Result source changed: {filename}')
    integration = read(RUN / 'java-resume-verification.json')
    if not integration['completed'] or not integration['cross_process_java_resume_verified']:
        raise ValueError('Resume verification missing')
    if integration['cash_absolute_error'] >= 1e-4 or integration['broker_orders_submitted']:
        raise ValueError('Replay accounting or order isolation failed')
    security = {}
    for relative in ('real_market_rl_trader/.env', '.runtime/credentials/kis-data-demo.json'):
        mode = stat.S_IMODE((WORKSPACE / relative).stat().st_mode)
        if mode != 0o600:
            raise ValueError(f'Private configuration has unexpected permissions: {relative}')
        security[relative] = {'mode': oct(mode)}
    subprocess.run(['git', 'check-ignore', '-q', '.env'], cwd=ROOT, check=True)
    for repository in ('real_market_rl_trader', 'rl-trading-core', 'rl_trader_data'):
        subprocess.run(['git', 'diff', '--check'], cwd=WORKSPACE / repository, check=True)
    future = read(RUN / 'future-holdout/availability.json')
    index = {
        'completed_local_checks': True, 'as_of': '2026-11-09', 'report_generated': False,
        'python': {'passed': int(python_result[1]), 'warnings': int(python_result[2]),
                   'seconds': float(python_result[3]), 'log': str(python_log.relative_to(WORKSPACE))},
        'java': {**java, 'boot_jar_built': True, 'log': str(java_log.relative_to(WORKSPACE))},
        'experiments': {'risk_models': 27, 'historical_test_symbol_rows': 135,
                        'predictive_models': 36, 'validation_policy_audits': 27,
                        'exploratory_stress_windows': 495},
        'data': {'duplicates_removed': completions['data-normalization']['duplicate_rows_removed'],
                 'canonical_observations_unchanged': completions['data-normalization']['canonical_observations_unchanged'],
                 'sparse_symbol_days_requeried': completions['data-recovery']['audited_defective_days'],
                 'qualified_days_recovered': completions['data-recovery']['quality_passed_days'],
                 'prices_fabricated': False, 'halt_classification_complete': False},
        'integration': integration, 'local_secret_permissions': security,
        'validation_feasible_conditions': assets['validation_feasible_conditions'],
        'profitability_or_risk_improvement_established': False,
        'future': {'ready': future['ready'], 'scores_computed': future['scores_computed'],
                   'earliest_evaluation_date': future['earliest_evaluation_date'],
                   'requires_future_data_collection': True, 'scheduler_installed': False},
        'remaining': ['Official adjudication of synchronous sparse intervals',
                      'Prospective frozen-policy evaluation after future data collection',
                      'Research evidence of consistent profitability or risk-budget feasibility'],
        'artifact_index': {
            'risk_comparison_figure': 'result-assets/frozen_comparison.png',
            'risk_comparison_vector': 'result-assets/frozen_comparison.svg',
            'risk_comparison_table': 'result-assets/frozen_test_summary.csv',
            'paired_seed_differences': 'result-assets/paired_seed_differences.csv',
            'sparse_session_stress_figure': 'result-assets/stress_comparison.png',
            'sparse_session_stress_table': 'result-assets/stress_by_symbol.csv',
            'predictive_skill': 'predictive-diagnostics/predictive_skill.png',
            'learning_risk_curves': 'global-policy-diagnostics/learning_and_risk.png',
            'value_function_audit': 'global-policy-diagnostics/critic_and_initial_policy.csv',
            'ppo_optimizer_statistics': 'global-policy-diagnostics/ppo_optimizer_metrics.csv',
            'action_counts_and_policy_probabilities': 'global-risk-study/frozen_evaluation/results.json',
            'earlier_cost_sensitivity': '../research/final-v2/cost_cross_evaluation/cost_sensitivity.png',
            'earlier_cost_sensitivity_table': '../research/final-v2/cost_cross_evaluation/metrics.csv',
            'earlier_synthetic_checks': '../research/final-v2/diagnostic_reproduction_check.json',
            'earlier_execution_ablations': '../research/final-v2/ablation/frozen_evaluation/metrics.csv',
            'prospective_reservation': 'future-holdout/reservation.json',
            'result_provenance': 'result-assets/manifest.json'},
        'interpretation': assets['limitations'],
        'verification_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for relative in index['artifact_index'].values():
        if not (RUN / relative).is_file():
            raise ValueError(f'Indexed artifact missing: {relative}')
    (RUN / 'verification.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'local_checks_passed': True, 'python': int(python_result[1]),
                      'java': java['tests'], 'report_generated': False}))


if __name__ == '__main__':
    verify()
