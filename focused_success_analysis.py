import os
import sys
import numpy as np
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd

# Default run directory - can be overridden via command line
BASE_DIR = "runs/run417"

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
    
    # Convert all numpy types to native Python types for JSON serialization
    for key in metrics:
        if isinstance(metrics[key], (np.integer, np.floating)):
            metrics[key] = float(metrics[key])
    
    return metrics

def analyze_run():
    """Analyze a single run with success-focused metrics."""
    
    run_name = os.path.basename(os.path.abspath(BASE_DIR))
    print(f"Processing {run_name}...")
    
    try:
        data = load_qd_data(BASE_DIR)
        if data is None:
            print(f"  ⚠️  Could not load data from {BASE_DIR}")
            return None
            
        data_dict, archive_data = data
        metrics = calculate_success_focused_metrics(data_dict, archive_data)
        
        print(f"  ✅ Archive: {metrics['total_successful_grasps']} successful, "
              f"{metrics['total_robust_grasps']} robust, "
              f"Success Rate: {metrics['success_rate']:.1f}%")
        
        return {
            'run_name': run_name,
            **metrics
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_metrics_visualization(metrics):
    """Create comprehensive visualization of all metrics."""
    
    # Create figure with 2x3 subplots for the 6 main metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define metrics to visualize
    metrics_to_plot = [
        ('success_rate', 'Success Rate\n(% of ALL Evaluations)', '#3498db', '%'),
        ('robust_success_rate', 'Robust Success Rate\n(% of ALL Evaluations)', '#2ecc71', '%'),
        ('total_successful_grasps', 'Total Successful Grasps\n(Archive Count)', '#9b59b6', ''),
        ('total_robust_grasps', 'Total Robust Grasps\n(Archive Count)', '#e74c3c', ''),
        ('qd_score', 'QD-Score\n(Sum of Fitness)', '#f39c12', ''),
        ('diversity_knn', 'Diversity\n(k-NN Distance)', '#1abc9c', '')
    ]
    
    for idx, (metric_key, title, color, suffix) in enumerate(metrics_to_plot):
        ax = axes[idx]
        value = metrics[metric_key]
        
        # Create a single bar
        bar = ax.bar([0], [value], color=color, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
        
        # Add value label on top of bar
        if suffix == '%':
            label = f'{value:.1f}%'
        elif metric_key in ['qd_score', 'total_successful_grasps', 'total_robust_grasps', 'n_evaluations']:
            label = f'{value:.0f}'
        else:
            label = f'{value:.4f}'
        
        ax.text(0, value, label, ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Style the subplot
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add a subtle background
        ax.set_facecolor('#f8f9fa')
    
    # Add overall title
    run_name = metrics.get('run_name', 'Unknown')
    fig.suptitle(f'Success-Focused QD Analysis: {run_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save the figure
    img_path = os.path.join(BASE_DIR, "success_focused_analysis.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return img_path

def print_metrics_summary(metrics):
    """Print comprehensive summary of all metrics."""
    
    print("\n" + "="*80)
    print("SUCCESS-FOCUSED ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n📊 SUCCESS METRICS:")
    print(f"  Total Successful Grasps (Archive): {metrics['total_successful_grasps']:.0f}")
    print(f"  Total Robust Grasps (Archive):     {metrics['total_robust_grasps']:.0f}")
    robust_pct = (metrics['total_robust_grasps'] / metrics['total_successful_grasps'] * 100) if metrics['total_successful_grasps'] > 0 else 0
    print(f"  Robust Percentage of Archive:      {robust_pct:.1f}%")
    
    print("\n📈 SUCCESS RATES (of all evaluations):")
    print(f"  Success Rate:                      {metrics['success_rate']:.1f}%")
    print(f"  Robust Success Rate:               {metrics['robust_success_rate']:.1f}%")
    
    print("\n🎯 QUALITY METRICS:")
    print(f"  QD-Score:                          {metrics['qd_score']:.0f}")
    print(f"  Average Fitness:                   {metrics['avg_fitness']:.3f}")
    print(f"  Max Fitness:                       {metrics['max_fitness']:.3f}")
    
    print("\n🌐 DIVERSITY METRICS:")
    print(f"  Archive Coverage:                  {metrics['success_archive_coverage']:.1f}%")
    print(f"  Diversity (k-NN Distance):         {metrics['diversity_knn']:.4f}")
    
    print("\n⏱️  EFFICIENCY:")
    print(f"  Total Evaluations:                 {metrics['n_evaluations']:.0f}")
    print(f"  Runtime:                           {metrics['runtime']:.1f} seconds")
    
    print("="*80)

def main():
    """Main analysis function."""
    # Allow command-line argument to override BASE_DIR
    global BASE_DIR
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
    
    print("🔍 Starting Success-Focused QD Analysis...")
    print("=" * 60)
    print(f"Analyzing run directory: {BASE_DIR}")
    print("=" * 60)
    
    # Analyze the run
    result = analyze_run()
    if result is None:
        print("\n❌ Failed to analyze run!")
        return
    
    # Print comprehensive metrics summary
    print_metrics_summary(result)
    
    # Create visualization
    print("\n📊 Creating metrics visualization...")
    img_path = create_metrics_visualization(result)
    print(f"[✔] Saved visualization: {img_path}")
    
    # Save detailed results to CSV
    csv_path = os.path.join(BASE_DIR, "success_focused_analysis.csv")
    df = pd.DataFrame([result])
    df.to_csv(csv_path, index=False)
    print(f"[✔] Saved CSV: {csv_path}")
    
    # Save detailed results to JSON for easy access
    json_path = os.path.join(BASE_DIR, "success_focused_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[✔] Saved JSON: {json_path}")
    
    print(f"\n✅ Success-focused analysis complete! Check {BASE_DIR}/ for results")

if __name__ == "__main__":
    main()