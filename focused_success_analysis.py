import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import environments.src.robots.mj_shadow_hand_consts as sh_consts
from run import JOINT_LOCKS

BASE_DIR = "ycb_chips_can_2025-10-08_18-27"
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
    
    metrics = {}
    
    # === SUCCESS-FOCUSED METRICS ===
    
    # 1. QD-Score: Sum of all fitness values in success archive
    metrics['qd_score'] = data_dict['success_archive_qd_score_hist'][-1]
    
    # 2. SUCCESS COUNTS (what you specifically asked for)
    metrics['total_successful_grasps'] = len(successful_fitnesses)  # Total in archive
    metrics['total_robust_grasps'] = len(robust_fitnesses)  # Robust subset
    
    # 3. SUCCESS RATES from all evaluations (not just archive)
    metrics['success_rate'] = data_dict['success_ratio_hist'][-1] * 100  # % of all evals
    
    # Calculate robust success rate from total evaluations
    # Estimate based on archive composition
    total_evals = data_dict['n_evals_hist'][-1]
    total_successful_evals = int(metrics['success_rate'] / 100 * total_evals)
    
    # Robust rate in archive
    robust_ratio_in_archive = len(robust_fitnesses) / len(successful_fitnesses) if len(successful_fitnesses) > 0 else 0
    estimated_total_robust_evals = int(total_successful_evals * robust_ratio_in_archive)
    metrics['robust_success_rate'] = (estimated_total_robust_evals / total_evals) * 100
    
    # 4. SPATIAL COVERAGE
    metrics['success_archive_coverage'] = data_dict['success_archive_cvg_hist'][-1] * 100
    
    # 5. Diversity
    metrics['diversity_knn'] = data_dict['success_archive_sparsity_3_hist'][-1]
    
    # 6. Quality metrics
    if len(successful_fitnesses) > 0:
        metrics['avg_fitness'] = np.mean(successful_fitnesses)
        metrics['max_fitness'] = np.max(successful_fitnesses)
    else:
        metrics['avg_fitness'] = 0
        metrics['max_fitness'] = 0
    
    # 7. Efficiency
    metrics['runtime'] = data_dict['run_time_hist'][-1]
    metrics['n_evaluations'] = data_dict['n_evals_hist'][-1]
    
    return metrics

def analyze_all_configurations():
    """Analyze all configurations with success-focused metrics."""
    
    all_folders = glob.glob(os.path.join(BASE_DIR, "*[0-9]"))
    
    results = []
    
    for folder in all_folders:
        folder_name = os.path.basename(folder)
        joint_lock, aa_config = extract_configs(folder_name)
        
        print(f"Processing {folder_name}...")
        
        try:
            data = load_qd_data(folder)
            if data is None:
                print(f"  ⚠️  Could not load data")
                continue
                
            data_dict, archive_data = data
            metrics = calculate_success_focused_metrics(data_dict, archive_data)
            
            result = {
                'folder': folder_name,
                'joint_lock': joint_lock,
                'aa_config': aa_config,
                **metrics
            }
            results.append(result)
            
            print(f"  ✅ Archive: {metrics['total_successful_grasps']} successful, "
                  f"{metrics['total_robust_grasps']} robust, "
                  f"Success Rate: {metrics['success_rate']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if not results:
        print("No valid results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by QD-Score
    df = df.sort_values('qd_score', ascending=False)
    
    print(f"\n📊 Successfully analyzed: {len(results)} configurations")
    
    return df

def create_success_focused_heatmaps(df):
    """Create heatmaps focused only on success metrics."""
    
    # Success-focused metrics only (removed spatial coverage as requested)
    metrics_to_plot = [
        ('success_rate', 'Success Rate (% of ALL Evaluations)', 'YlGnBu'),
        ('robust_success_rate', 'Robust Success Rate (% of ALL Evaluations)', 'Greens'),
        ('total_successful_grasps', 'Total Successful Grasps (Archive Count)', 'Blues'),
        ('total_robust_grasps', 'Total Robust Grasps (Archive Count)', 'Reds'),
        ('qd_score', 'QD-Score (Sum of Fitness)', 'viridis'),
        # ('diversity_knn', 'Diversity (k-NN Distance)', 'Oranges')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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
        if metric in ['qd_score', 'total_successful_grasps', 'total_robust_grasps']:
            fmt = ".0f"  # No decimals for counts
        elif metric in ['success_rate', 'robust_success_rate']:
            fmt = ".1f"  # One decimal for percentages
        else:
            fmt = ".3f"  # Three decimals for diversity
            
        # Create heatmap with proper formatting
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, 
                   linewidths=0.5, ax=ax, cbar_kws={'label': metric})
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylabel("Joint Lock Configuration")
        ax.set_xlabel("AA Configuration")
        
        # Rotate labels for readability
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save heatmap
    img_path = os.path.join(BASE_DIR, "success_focused_analysis.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✔] Saved success-focused analysis: {img_path}")
    
    return img_path

def print_robust_grasp_summary(df):
    """Print summary of robust grasp performance."""
    
    print("\n" + "="*80)
    print("ROBUST GRASP ANALYSIS SUMMARY")
    print("="*80)
    
    # Find best and worst for robust grasps
    best_robust_count = df.loc[df['total_robust_grasps'].idxmax()]
    worst_robust_count = df.loc[df['total_robust_grasps'].idxmin()]
    
    best_robust_rate = df.loc[df['robust_success_rate'].idxmax()]
    worst_robust_rate = df.loc[df['robust_success_rate'].idxmin()]
    
    print(f"BEST ROBUST GRASP COUNT:")
    print(f"  {best_robust_count['joint_lock']} + {best_robust_count['aa_config']}: "
          f"{best_robust_count['total_robust_grasps']:.0f} robust grasps "
          f"out of {best_robust_count['total_successful_grasps']:.0f} total")
    
    print(f"\nWORST ROBUST GRASP COUNT:")
    print(f"  {worst_robust_count['joint_lock']} + {worst_robust_count['aa_config']}: "
          f"{worst_robust_count['total_robust_grasps']:.0f} robust grasps "
          f"out of {worst_robust_count['total_successful_grasps']:.0f} total")
    
    print(f"\nBEST ROBUST SUCCESS RATE:")
    print(f"  {best_robust_rate['joint_lock']} + {best_robust_rate['aa_config']}: "
          f"{best_robust_rate['robust_success_rate']:.1f}% of all evaluations")
    
    print(f"\nWORST ROBUST SUCCESS RATE:")
    print(f"  {worst_robust_rate['joint_lock']} + {worst_robust_rate['aa_config']}: "
          f"{worst_robust_rate['robust_success_rate']:.1f}% of all evaluations")

def main():
    """Main analysis function."""
    print("🔍 Starting Success-Focused QD Analysis...")
    print("=" * 60)
    
    # Analyze all configurations
    df = analyze_all_configurations()
    if df is None:
        return
    
    # Save detailed results
    # csv_path = os.path.join(BASE_DIR, "success_focused_analysis.csv")
    # df.to_csv(csv_path, index=False)
    # print(f"[✔] Saved detailed CSV: {csv_path}")
    
    # Create success-focused heatmaps
    create_success_focused_heatmaps(df)
    
    # Print robust grasp summary
    print_robust_grasp_summary(df)
    
    # Print top performers by different metrics
    print(f"\n🏆 TOP 5 BY QD-SCORE:")
    print("-" * 90)
    top_qd = df.head(5)
    print(f"{'Rank':<4} {'Configuration':<25} {'AA':<8} {'QD-Score':<8} {'Success':<8} {'Robust':<7} {'Total'}")
    print("-" * 90)
    for i, (_, row) in enumerate(top_qd.iterrows()):
        print(f"{i+1:4d} {row['joint_lock']:<25} {row['aa_config']:<8} "
              f"{row['qd_score']:8.0f} {row['success_rate']:7.1f}% "
              f"{row['total_robust_grasps']:6.0f} {row['total_successful_grasps']:5.0f}")
    
    print(f"\n🏆 TOP 5 BY ROBUST GRASP COUNT:")
    print("-" * 90)
    top_robust = df.nlargest(5, 'total_robust_grasps')
    for i, (_, row) in enumerate(top_robust.iterrows()):
        robust_percentage = (row['total_robust_grasps'] / row['total_successful_grasps']) * 100
        print(f"{i+1:4d} {row['joint_lock']:<25} {row['aa_config']:<8} "
              f"{row['total_robust_grasps']:6.0f} {robust_percentage:6.1f}% "
              f"{row['total_successful_grasps']:5.0f}")
    
    print(f"\n✅ Success-focused analysis complete! Check {BASE_DIR}/ for results")

if __name__ == "__main__":
    main()