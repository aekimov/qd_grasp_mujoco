import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime
import environments.src.robots.mj_shadow_hand_consts as sh_consts
from run import JOINT_LOCKS

FOLDER_NAMES = ['ycb_chips_can_2025-11-30_15-53']

AA_CONFIGS = list(sh_consts.AA_CONFIGURATIONS.keys())

def extract_configs(folder_name):
    """Extract joint lock and AA configuration from folder name."""
    folder_name = folder_name.rstrip("0123456789")  # remove trailing run index
    
    for joint in sorted(JOINT_LOCKS, key=len, reverse=True):
        if folder_name.startswith(joint):
            rest = folder_name[len(joint):].strip("_")
            aa = rest if rest in AA_CONFIGS else "default"
            return joint, aa
    return "unknown", "default"

def load_qd_data(folder_path):
    """Load QD data from folder."""
    folder = Path(folder_path)
    
    # Load QD metrics from data_export.pkl
    data_export_path = folder / "data_export.pkl"
    if not data_export_path.exists():
        return None
        
    with open(data_export_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Load final archive from npz file
    success_archives_dir = folder / "success_archives"
    if not success_archives_dir.exists():
        return None
        
    npz_files = list(success_archives_dir.glob("individuals_*.npz"))
    if not npz_files:
        return None
        
    # Get the latest npz file
    latest_npz = max(npz_files, key=lambda x: int(x.stem.split('_')[-1]))
    archive_data = np.load(latest_npz, allow_pickle=True)
    
    return data_dict, archive_data

def calculate_success_focused_metrics(data_dict, archive_data):
    """Calculate metrics focused only on success rates and robust grasps."""
    
    # Extract archive data
    fitnesses = archive_data['fitnesses'] 
    infos = archive_data['infos']
    info_keys = archive_data['infos_keys']
    
    # Get success flags from archive
    success_idx = np.where(info_keys == 'is_success')[0][0]
    robust_idx = np.where(info_keys == 'is_robust_grasp')[0][0]
    
    is_success = infos[:, success_idx].astype(bool)
    is_robust = infos[:, robust_idx].astype(bool)
    
    # All entries in success archive should be successful (sanity check)
    successful_fitnesses = fitnesses[is_success]
    robust_fitnesses = fitnesses[is_robust]
    
    # Extract ALL robust grasps from robust_grasps field
    all_robust_grasps_count = 0
    if 'robust_grasps' in archive_data:
        robust_grasps_array = archive_data['robust_grasps']
        if robust_grasps_array is not None and len(robust_grasps_array) > 0:
            all_robust_grasps_count = len(robust_grasps_array)
    
    metrics = {}
    
    # === SUCCESS-FOCUSED METRICS ===
    
    # 1. QD-Score: Sum of all fitness values in success archive
    metrics['qd_score'] = data_dict['success_archive_qd_score_hist'][-1]
    
    # 2. SUCCESS COUNTS (what you specifically asked for)
    metrics['total_successful_grasps'] = len(successful_fitnesses)  # Total in archive
    metrics['total_robust_grasps'] = len(robust_fitnesses)  # Robust subset
    
    # 3. ALL ROBUST GRASPS COUNT (from robust_grasps field - all evaluations)
    metrics['all_robust_grasps_count'] = all_robust_grasps_count
    
    # 4. SUCCESS RATES from all evaluations (not just archive)
    metrics['success_rate'] = data_dict['success_ratio_hist'][-1] * 100  # % of all evals
    
    # Calculate robust success rate from total evaluations
    # Estimate based on archive composition
    total_evals = data_dict['n_evals_hist'][-1]
    total_successful_evals = int(metrics['success_rate'] / 100 * total_evals)
    
    # Robust rate in archive
    robust_ratio_in_archive = len(robust_fitnesses) / len(successful_fitnesses) if len(successful_fitnesses) > 0 else 0
    estimated_total_robust_evals = int(total_successful_evals * robust_ratio_in_archive)
    metrics['robust_success_rate'] = (estimated_total_robust_evals / total_evals) * 100
    
    # 5. SPATIAL COVERAGE
    metrics['success_archive_coverage'] = data_dict['success_archive_cvg_hist'][-1] * 100
    
    # 6. Diversity
    metrics['diversity_knn'] = data_dict['success_archive_sparsity_3_hist'][-1]
    
    # 7. Quality metrics
    if len(successful_fitnesses) > 0:
        metrics['avg_fitness'] = np.mean(successful_fitnesses)
        metrics['max_fitness'] = np.max(successful_fitnesses)
    else:
        metrics['avg_fitness'] = 0
        metrics['max_fitness'] = 0
    
    # 8. Efficiency
    metrics['runtime'] = data_dict['run_time_hist'][-1]
    metrics['n_evaluations'] = data_dict['n_evals_hist'][-1]
    
    return metrics

def analyze_all_configurations():
    """Analyze all configurations across multiple folders with averaging."""
    
    # Store results for each folder
    all_folder_results = {}
    
    print(f"📊 Processing {len(FOLDER_NAMES)} base folders...")
    print("=" * 80)
    
    # Process each base folder
    for base_folder in FOLDER_NAMES:
        print(f"\n📁 Processing base folder: {base_folder}")
        
        if not os.path.exists(base_folder):
            print(f"  ⚠️  Folder not found: {base_folder}")
            continue
        
        all_subfolders = glob.glob(os.path.join(base_folder, "*[0-9]"))
        folder_results = []
        
        for folder in all_subfolders:
            folder_name = os.path.basename(folder)
            joint_lock, aa_config = extract_configs(folder_name)
            
            try:
                data = load_qd_data(folder)
                if data is None:
                    continue
                    
                data_dict, archive_data = data
                metrics = calculate_success_focused_metrics(data_dict, archive_data)
                
                result = {
                    'folder': folder_name,
                    'joint_lock': joint_lock,
                    'aa_config': aa_config,
                    **metrics
                }
                folder_results.append(result)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {folder_name}: {e}")
                continue
        
        all_folder_results[base_folder] = folder_results
        print(f"  ✅ Loaded {len(folder_results)} configurations from {base_folder}")
    
    # Combine and average results
    print("\n" + "=" * 80)
    print("🔄 Calculating averages across folders...")
    print("=" * 80)
    
    # Create a dictionary to store all runs for each configuration
    config_runs = {}
    
    for base_folder, results in all_folder_results.items():
        for result in results:
            config_key = (result['joint_lock'], result['aa_config'])
            if config_key not in config_runs:
                config_runs[config_key] = []
            config_runs[config_key].append(result)
    
    # Calculate averages and filter configurations that exist in all folders
    averaged_results = []
    
    for config_key, runs in config_runs.items():
        joint_lock, aa_config = config_key
        
        # Sanity check: only include if configuration exists in all folders
        if len(runs) != len(FOLDER_NAMES):
            print(f"  ⚠️  Skipping {joint_lock} + {aa_config}: "
                  f"found in {len(runs)}/{len(FOLDER_NAMES)} folders")
            continue
        
        # Calculate averages for all numeric metrics
        avg_result = {
            'joint_lock': joint_lock,
            'aa_config': aa_config,
            'n_runs': len(runs)
        }
        
        # List of metrics to average
        metrics_to_average = [
            'qd_score', 'total_successful_grasps', 'total_robust_grasps', 'all_robust_grasps_count',
            'success_rate', 'robust_success_rate', 'success_archive_coverage',
            'diversity_knn', 'avg_fitness', 'max_fitness', 'runtime', 'n_evaluations'
        ]
        
        for metric in metrics_to_average:
            values = [run[metric] for run in runs]
            avg_result[metric] = np.mean(values)
            avg_result[f'{metric}_std'] = np.std(values)
        
        averaged_results.append(avg_result)
        
        print(f"  ✅ {joint_lock} + {aa_config}: "
              f"averaged {len(runs)} runs - "
              f"QD-Score: {avg_result['qd_score']:.0f} ± {avg_result['qd_score_std']:.0f}")
    
    if not averaged_results:
        print("❌ No valid averaged results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(averaged_results)
    
    # Sort by QD-Score
    df = df.sort_values('qd_score', ascending=False)
    
    print(f"\n📊 Successfully averaged: {len(averaged_results)} configurations")
    print(f"   (Each based on {len(FOLDER_NAMES)} folder runs)")
    
    return df

def create_success_focused_heatmaps(df, output_dir='.', timestamp_str=''):
    """Create heatmaps focused only on success metrics."""
    
    # Success-focused metrics only
    metrics_to_plot = [
        ('total_successful_grasps', 'Total Successful Grasps (Archive Count - Averaged)', 'Blues'),
        ('total_robust_grasps', 'Total Robust Grasps (Archive Count - Averaged)', 'Reds'),
        ('qd_score', 'QD-Score (Sum of Fitness - Averaged)', 'viridis'),
        ('all_robust_grasps_count', 'All Robust Grasps Count (All Evaluations - Averaged)', 'Oranges'),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes = axes.flatten()
    
    for idx, (metric, title, cmap) in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
            
        # Create pivot table
        pivot = df.pivot(index="joint_lock", columns="aa_config", values=metric)
        pivot = pivot.reindex(index=JOINT_LOCKS, columns=AA_CONFIGS).fillna(0)
        
        # Plot heatmap
        ax = axes[idx]
        
        # Format annotation based on metric type
        if metric in ['qd_score', 'total_successful_grasps', 'total_robust_grasps', 'all_robust_grasps_count']:
            fmt = ".0f"  # No decimals for counts
        elif metric in ['success_rate', 'robust_success_rate']:
            fmt = ".1f"  # One decimal for percentages
        else:
            fmt = ".3f"  # Three decimals for diversity
            
        # Create heatmap with proper formatting
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, 
                   linewidths=0.5, ax=ax, cbar_kws={'label': metric})
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel("Joint Lock Configuration")
        ax.set_xlabel("AA Configuration")
        
        # Rotate labels for readability
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save heatmap with timestamp
    if timestamp_str:
        img_path = os.path.join(output_dir, f"averaged_success_analysis_{timestamp_str}.png")
    else:
        img_path = os.path.join(output_dir, "averaged_success_analysis.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved averaged success analysis: {img_path}")
    
    return img_path

def print_robust_grasp_summary(df):
    """Print summary of robust grasp performance."""
    
    print("\n" + "="*80)
    print("ROBUST GRASP ANALYSIS SUMMARY (AVERAGED ACROSS FOLDERS)")
    print("="*80)
    
    # Find best and worst for robust grasps
    best_robust_count = df.loc[df['total_robust_grasps'].idxmax()]
    worst_robust_count = df.loc[df['total_robust_grasps'].idxmin()]
    
    best_robust_rate = df.loc[df['robust_success_rate'].idxmax()]
    worst_robust_rate = df.loc[df['robust_success_rate'].idxmin()]
    
    print(f"BEST ROBUST GRASP COUNT:")
    print(f"  {best_robust_count['joint_lock']} + {best_robust_count['aa_config']}: "
          f"{best_robust_count['total_robust_grasps']:.0f} ± {best_robust_count['total_robust_grasps_std']:.0f} robust grasps "
          f"out of {best_robust_count['total_successful_grasps']:.0f} ± {best_robust_count['total_successful_grasps_std']:.0f} total")
    
    print(f"\nWORST ROBUST GRASP COUNT:")
    print(f"  {worst_robust_count['joint_lock']} + {worst_robust_count['aa_config']}: "
          f"{worst_robust_count['total_robust_grasps']:.0f} ± {worst_robust_count['total_robust_grasps_std']:.0f} robust grasps "
          f"out of {worst_robust_count['total_successful_grasps']:.0f} ± {worst_robust_count['total_successful_grasps_std']:.0f} total")
    
    print(f"\nBEST ROBUST SUCCESS RATE:")
    print(f"  {best_robust_rate['joint_lock']} + {best_robust_rate['aa_config']}: "
          f"{best_robust_rate['robust_success_rate']:.1f}% ± {best_robust_rate['robust_success_rate_std']:.1f}%")
    
    print(f"\nWORST ROBUST SUCCESS RATE:")
    print(f"  {worst_robust_rate['joint_lock']} + {worst_robust_rate['aa_config']}: "
          f"{worst_robust_rate['robust_success_rate']:.1f}% ± {worst_robust_rate['robust_success_rate_std']:.1f}%")

def main():
    """Main analysis function."""
    print("🔍 Starting Success-Focused QD Analysis (Multi-Folder Averaging)...")
    print("=" * 80)
    
    # Generate timestamp for unique filenames
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Analyze all configurations
    df = analyze_all_configurations()
    if df is None:
        return
    
    # Create output directory
    output_dir = "averaged_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results with timestamp
    csv_path = os.path.join(output_dir, f"averaged_success_analysis_{timestamp_str}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Saved detailed CSV: {csv_path}")
    
    # Create success-focused heatmaps with timestamp
    create_success_focused_heatmaps(df, output_dir, timestamp_str)
    
    # Print robust grasp summary
    print_robust_grasp_summary(df)
    
    # Print top performers by different metrics
    print(f"\n🏆 TOP 5 BY QD-SCORE (AVERAGED):")
    print("-" * 100)
    top_qd = df.head(5)
    print(f"{'Rank':<4} {'Configuration':<25} {'AA':<8} {'QD-Score':<15} {'Success':<12} {'Robust':<10} {'Total'}")
    print("-" * 100)
    for i, (_, row) in enumerate(top_qd.iterrows()):
        print(f"{i+1:4d} {row['joint_lock']:<25} {row['aa_config']:<8} "
              f"{row['qd_score']:7.0f}±{row['qd_score_std']:4.0f} "
              f"{row['success_rate']:6.1f}±{row['success_rate_std']:3.1f}% "
              f"{row['total_robust_grasps']:5.0f}±{row['total_robust_grasps_std']:3.0f} "
              f"{row['total_successful_grasps']:5.0f}±{row['total_successful_grasps_std']:3.0f}")
    
    print(f"\n🏆 TOP 5 BY ROBUST GRASP COUNT (AVERAGED):")
    print("-" * 100)
    top_robust = df.nlargest(5, 'total_robust_grasps')
    for i, (_, row) in enumerate(top_robust.iterrows()):
        robust_percentage = (row['total_robust_grasps'] / row['total_successful_grasps']) * 100
        print(f"{i+1:4d} {row['joint_lock']:<25} {row['aa_config']:<8} "
              f"{row['total_robust_grasps']:5.0f}±{row['total_robust_grasps_std']:3.0f} "
              f"{robust_percentage:6.1f}% "
              f"{row['total_successful_grasps']:5.0f}±{row['total_successful_grasps_std']:3.0f}")
    
    print(f"\n✅ Averaged analysis complete! Check {output_dir}/ for results")

if __name__ == "__main__":
    main()