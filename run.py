import subprocess
import os
import glob
import matplotlib.pyplot as plt
import environments.src.robots.mj_shadow_hand_consts as sh_consts
from datetime import datetime

# Configurations to test
JOINT_LOCKS = [
    "none",

    # Single finger locking
    "lock_index",
    "lock_middle",
    "lock_ring",
    "lock_little",
    # "lock_thumb",

    # Two finger locking
    "lock_ring_little",
    "lock_middle_little",    
    "lock_middle_ring",
    "lock_index_little",
    "lock_index_ring",
    "lock_index_middle",

    # Three finger locking
    "lock_middle_ring_little",
    "lock_index_ring_little",
    "lock_index_middle_little",
    "lock_index_middle_ring"
]

AA_CONFIGS = [key for key in sh_consts.AA_CONFIGURATIONS.keys()] # if key != 'default'

# Experiment settings
NBR = 20000
OBJ = "ycb_chips_can"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
LOG_PATH = f"{OBJ}_{timestamp}"
os.makedirs(LOG_PATH, exist_ok=True)

def run_test(joint_lock, aa_config=None):
    folder_parts = [joint_lock]
    cmd = [
        "python", "run_qd_grasp.py",
        "-a", "contact_cma_mae",
        "-r", "shadow",
        "-nbr", str(NBR),
        "-o", OBJ,
        "-jl", joint_lock,
        "-l", LOG_PATH,
        "-p", str(500),
        "-ll"
    ]

    if aa_config:
        cmd.extend(["--aa-config", aa_config])
        folder_parts.append(aa_config)

    folder_name = "_".join(folder_parts)
    cmd.extend(["-f", folder_name])

    print(f"\n[Running test: {folder_name}]")
    subprocess.run(cmd)

def find_latest_run_folder(joint_lock):
    prefix = f"test_{joint_lock}"
    pattern = os.path.join(LOG_PATH, f"{prefix}[0-9]*")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

def parse_run_infos(path):
    n_success = n_eval = None
    try:
        with open(path, "r") as f:
            for line in f:
                if "Number of successful individuals" in line:
                    n_success = int(line.split(":")[-1].strip())
                elif "Number of evaluations (progression_monitoring.n_eval)" in line:
                    n_eval = int(line.split(":")[-1].strip())
    except Exception:
        pass
    return n_success, n_eval

def read_stats(run_folder):
    path = os.path.join(run_folder, "run_infos.yaml")
    if not os.path.exists(path):
        print(f"[!] Missing run_infos.yaml in {run_folder}")
        return None, None
    return parse_run_infos(path)

def save_txt(results, path):
    with open(path, "w") as f:
        f.write("=== Grasping Results Summary ===\n")
        for name, success, total, rate in results:
            f.write(f"{name:<30} {success}/{total} successful grasps ({rate:.1f}%)\n")
    print(f"Saved summary to: {path}")

def save_plot(results, path):
    """
    Draws a line plot with labeled points from experiment results.

    Parameters:
        results: List of tuples (label, n_success, n_total, success_rate)
        path: Path to save the plot image
    """
    names = [name for name, _, _, _ in results]
    rates = [rate for _, _, _, rate in results]

    plt.figure(figsize=(12, 6))
    plt.plot(names, rates, marker='x', markersize=10, linewidth=2, color='brown', label='Success Rate')

    # Annotate each point with its value
    for i, (x, y) in enumerate(zip(names, rates)):
        plt.text(i, y + 0.3, f"{y:.2f}", ha='center', fontsize=9, color='brown')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Grasping Success Rate per Joint Lock Configuration")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[✔] Line plot saved to: {path}")

def main():
    results = []

    total_runs = len(JOINT_LOCKS) * (len(AA_CONFIGS) if AA_CONFIGS else 1)
    current_run = 0

    print(f"\n{'='*60}")
    print(f"Starting {total_runs} experiments:")
    print(f"  - {len(JOINT_LOCKS)} joint lock configurations")
    print(f"  - {len(AA_CONFIGS) if AA_CONFIGS else 1} AA configurations")
    print(f"{'='*60}\n")

    for joint_lock in JOINT_LOCKS:
        if AA_CONFIGS:
            for aa_config in AA_CONFIGS:
                current_run += 1
                print(f"\n{'='*60}")
                print(f"Progress: {current_run}/{total_runs}")
                print(f"{'='*60}")
                run_test(joint_lock, aa_config)
        else:
            current_run += 1
            print(f"\n{'='*60}")
            print(f"Progress: {current_run}/{total_runs}")
            print(f"{'='*60}")
            run_test(joint_lock)

    # Collect all folders matching naming pattern
    all_run_folders = glob.glob(os.path.join(LOG_PATH, "*[0-9]"))

    for folder in all_run_folders:
        folder_name = os.path.basename(folder)

        # Read stats
        success, total = read_stats(folder)
        if success is not None and total is not None:
            rate = (success / total) * 100
            results.append((folder_name, success, total, rate))

    # Print summary
    print("\n=== Grasping Results Summary ===")
    for name, success, total, rate in results:
        print(f"{name:<40} {success}/{total} successful grasps ({rate:.1f}%)")

    # Save outputs
    # summary_path = os.path.join(LOG_PATH, "summary.txt")
    plot_path = os.path.join(LOG_PATH, "success_plot.png")
    # save_txt(results, summary_path)
    save_plot(results, plot_path)

if __name__ == "__main__":
    main()
