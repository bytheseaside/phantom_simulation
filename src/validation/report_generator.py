"""
report_generator.py

Generate Markdown and CSV reports summarizing validation results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def generate_summary_csv(
    all_results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Generate summary_all.csv with all S matrices ranked by Solver vs F_subset RMSE.
    
    Columns: algorithm, s_matrix_name, solver_vs_subset_rmse, solver_vs_full_rmse,
             full_vs_subset_rmse, max_error, mean_error, relative_rmse
    """
    rows = []
    for result in all_results:
        rows.append({
            'algorithm': result['algorithm'],
            's_matrix_name': result['s_matrix_name'],
            'solver_vs_subset_rmse': result['solver_vs_subset_rmse'],
            'solver_vs_full_rmse': result['solver_vs_full_rmse'],
            'full_vs_subset_rmse': result['full_vs_subset_rmse'],
            'solver_vs_subset_max_error': result['solver_vs_subset_max_error'],
            'solver_vs_subset_mean_error': result['solver_vs_subset_mean_error'],
            'solver_vs_subset_relative_rmse': result['solver_vs_subset_relative_rmse'],
            'n_selected_dipoles': result['n_selected_dipoles'],
            'n_test_cases': result['n_test_cases']
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by solver_vs_subset_rmse (ascending = better)
    df = df.sort_values('solver_vs_subset_rmse', ascending=True)
    
    # Save with full precision
    df.to_csv(output_path, index=False, float_format='%.15e')
    print(f"  Created summary CSV: {output_path.name}")


def generate_s_matrix_report(
    s_matrix_name: str,
    algorithm_name: str,
    result_data: Dict[str, Any],
    output_path: Path
):
    """
    Generate Markdown report for a single S matrix.
    
    Includes:
    - Overall metrics
    - Regional breakdown
    - Per-case summary
    - Key findings
    """
    
    with open(output_path, 'w') as f:
        # Header
        f.write(f"# Validation Report: {s_matrix_name}\n\n")
        f.write(f"**Algorithm:** {algorithm_name}\n\n")
        f.write(f"**Selection:** {result_data['n_selected_dipoles']} / 36 dipoles\n\n")
        f.write(f"**Test Cases:** {result_data['n_test_cases']}\n\n")
        f.write("---\n\n")
        
        # Overall metrics
        f.write("## Overall Performance\n\n")
        f.write("### RMSE Comparison (Mean across all test cases)\n\n")
        f.write("| Comparison | RMSE (μV) | Max Error (μV) | Mean Error (μV) | Relative RMSE (%) |\n")
        f.write("|------------|-----------|----------------|-----------------|-------------------|\n")
        
        sv_full = result_data['solver_vs_full_rmse'] * 1e6
        sv_subset = result_data['solver_vs_subset_rmse'] * 1e6
        f_subset = result_data['full_vs_subset_rmse'] * 1e6
        
        sv_full_max = result_data['solver_vs_full_max_error'] * 1e6
        sv_subset_max = result_data['solver_vs_subset_max_error'] * 1e6
        
        sv_full_mean = result_data['solver_vs_full_mean_error'] * 1e6
        sv_subset_mean = result_data['solver_vs_subset_mean_error'] * 1e6
        
        sv_full_rel = result_data['solver_vs_full_relative_rmse'] * 100
        sv_subset_rel = result_data['solver_vs_subset_relative_rmse'] * 100
        
        f.write(f"| Solver vs F_full | {sv_full:.2f} | {sv_full_max:.2f} | {sv_full_mean:.2f} | {sv_full_rel:.2f} |\n")
        f.write(f"| Solver vs F_subset | {sv_subset:.2f} | {sv_subset_max:.2f} | {sv_subset_mean:.2f} | {sv_subset_rel:.2f} |\n")
        f.write(f"| F_full vs F_subset | {f_subset:.2f} | - | - | - |\n\n")
        
        # Interpretation
        f.write("### Interpretation\n\n")
        
        if sv_full < 1.0:
            f.write(f"✅ **F_full accuracy:** Excellent (RMSE = {sv_full:.2f} μV)\n\n")
        elif sv_full < 5.0:
            f.write(f"✅ **F_full accuracy:** Good (RMSE = {sv_full:.2f} μV)\n\n")
        else:
            f.write(f"⚠️ **F_full accuracy:** Moderate (RMSE = {sv_full:.2f} μV)\n\n")
        
        degradation = ((sv_subset - sv_full) / sv_full * 100) if sv_full > 0 else 0
        
        if degradation < 10:
            f.write(f"✅ **Selection quality:** Excellent (degradation = {degradation:.1f}%)\n\n")
        elif degradation < 30:
            f.write(f"✅ **Selection quality:** Good (degradation = {degradation:.1f}%)\n\n")
        elif degradation < 50:
            f.write(f"⚠️ **Selection quality:** Moderate (degradation = {degradation:.1f}%)\n\n")
        else:
            f.write(f"❌ **Selection quality:** Poor (degradation = {degradation:.1f}%)\n\n")
        
        f.write("---\n\n")
        
        # Regional analysis
        if 'regional_breakdown' in result_data and result_data['regional_breakdown']:
            f.write("## Regional Breakdown\n\n")
            f.write("Mean errors by brain region (averaged across all test cases):\n\n")
            f.write("| Region | RMSE (μV) | Max Error (μV) | Mean Error (μV) |\n")
            f.write("|--------|-----------|----------------|------------------|\n")
            
            # Compute regional averages
            regional_avg = {}
            for case_regional in result_data['regional_breakdown']:
                for region, metrics in case_regional.items():
                    if region not in regional_avg:
                        regional_avg[region] = {'rmse': [], 'max': [], 'mean': []}
                    regional_avg[region]['rmse'].append(metrics['rmse'])
                    regional_avg[region]['max'].append(metrics['max_error'])
                    regional_avg[region]['mean'].append(metrics['mean_error'])
            
            for region in sorted(regional_avg.keys()):
                rmse = np.mean(regional_avg[region]['rmse']) * 1e6
                max_err = np.mean(regional_avg[region]['max']) * 1e6
                mean_err = np.mean(regional_avg[region]['mean']) * 1e6
                f.write(f"| {region} | {rmse:.2f} | {max_err:.2f} | {mean_err:.2f} |\n")
            
            f.write("\n---\n\n")
        
        # Per-case summary
        f.write("## Per-Case Summary\n\n")
        f.write("| Test Case | Solver vs F_subset RMSE (μV) | Max Error (μV) |\n")
        f.write("|-----------|-------------------------------|----------------|\n")
        
        for case_result in result_data['per_case_results']:
            case_name = case_result['case_name']
            rmse = case_result['solver_vs_subset']['rmse'] * 1e6
            max_err = case_result['solver_vs_subset']['max_error'] * 1e6
            f.write(f"| {case_name} | {rmse:.2f} | {max_err:.2f} |\n")
        
        f.write("\n---\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if sv_subset < 5.0 and degradation < 20:
            f.write("✅ **This selection is recommended for practical use.**\n\n")
            f.write("The reconstruction accuracy is excellent and the selected dipoles capture most of the forward model information.\n\n")
        elif sv_subset < 10.0 and degradation < 40:
            f.write("⚠️ **This selection can be used with caution.**\n\n")
            f.write("Reconstruction accuracy is acceptable but consider validating on additional test cases specific to your application.\n\n")
        else:
            f.write("❌ **This selection may not be suitable for high-precision applications.**\n\n")
            f.write("Consider:\n")
            f.write("- Selecting more dipoles\n")
            f.write("- Using a different selection algorithm\n")
            f.write("- Reviewing the forward model accuracy (F_full)\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated from validation results*\n")
    
    print(f"  Created Markdown report: {output_path.name}")


def generate_case_detail_csv(
    case_name: str,
    probe_names: List[str],
    regions: List[str],
    V_solver: np.ndarray,
    V_full: np.ndarray,
    V_subset: np.ndarray,
    output_path: Path
):
    """
    Generate detailed CSV for a single test case.
    
    Columns: probe, region, V_solver, V_full, V_subset,
             error_solver_full, error_solver_subset, error_full_subset
    """
    rows = []
    for i, (probe, region) in enumerate(zip(probe_names, regions)):
        rows.append({
            'probe': probe,
            'region': region,
            'V_solver': V_solver[i],
            'V_full': V_full[i],
            'V_subset': V_subset[i],
            'error_solver_full': V_solver[i] - V_full[i],
            'error_solver_subset': V_solver[i] - V_subset[i],
            'error_full_subset': V_full[i] - V_subset[i]
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.15e')


def generate_combined_test_cases_csv(
    all_case_data: List[Dict[str, Any]],
    output_path: Path
):
    """
    Combine all test case data into a single CSV for programmatic analysis.
    
    Columns: s_matrix, case, probe, region, V_solver, V_full, V_subset, errors
    """
    rows = []
    for case_data in all_case_data:
        s_matrix = case_data['s_matrix_name']
        case_name = case_data['case_name']
        
        for i, probe in enumerate(case_data['probe_names']):
            rows.append({
                's_matrix': s_matrix,
                'case': case_name,
                'probe': probe,
                'region': case_data['regions'][i],
                'V_solver': case_data['V_solver'][i],
                'V_full': case_data['V_full'][i],
                'V_subset': case_data['V_subset'][i],
                'error_solver_full': case_data['V_solver'][i] - case_data['V_full'][i],
                'error_solver_subset': case_data['V_solver'][i] - case_data['V_subset'][i],
                'error_full_subset': case_data['V_full'][i] - case_data['V_subset'][i]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.15e')
    print(f"  Created combined CSV: {output_path.name}")
