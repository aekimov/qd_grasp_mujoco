# Analysis: Why Robust Grasps Fail During Replay

## Problem Statement
Grasps marked as `is_robust_grasp=1.0` (fitness=2.0) in the archive sometimes fail when replayed using `replay_grasps.py`.

## Complete Pipeline Trace

### 1. **During Training Evaluation** (`algorithms/evaluate.py` + `environments/src/mj_robot_grasping.py`)

```
evaluate_grasp_ind()
  └─> evaluate_6dof_pose()
      ├─> env.reset()
      ├─> env.set_6dof_gripper_pose(gripper_6dof_pose)
      ├─> env.close_gripper()
      │   └─> After closing: wait 200 steps (SETTLING_PARAMETERS['after_gripper_close'])
      │
      └─> env.apply_all_gripper_shaking(gripper_6dof_output_data)  # LINE 134 in evaluate.py
          ├─> For each shake axis (joints [2, 5]):
          │   ├─> apply_gripper_shaking(joint_index, gripper_joint_infos)
          │   │   └─> For each shake position [+target, -target, 0]:
          │   │       ├─> command_joint_pose(target_position, ...)
          │   │       │   └─> Execute shake for 68 steps (t_cmd_stable)
          │   │       │
          │   │       └─> Check: is_grasping_candidate() after EACH position
          │   │           └─> If NO contact: return is_being_grasped=False immediately
          │   │
          │   └─> Between shake axes: wait 50 steps (SETTLING_PARAMETERS['between_shake_axes'])
          │
          └─> After ALL shakes successful:
              ├─> Set gripper_6dof_output_data['is_robust_grasp'] = True
              ├─> Set gripper_6dof_output_data['is_success'] = True  
              └─> Calculate fitness = 2.0
```

### 2. **Archive Storage Logic** (`algorithms/archives/elite_structured_archive.py`)

```python
# Lines 46-51: When BOTH candidate and niche are successful
if candidate_is_scs and niche_is_scs:
    candidate_fitness = ind_fit
    niche_fitness = self._map_fits[key_map_ids]
    if candidate_fitness > niche_fitness:  # ← CRITICAL: Only replaces if FITNESS IS HIGHER
        self._set_ind(...)
```

**Key Point**: The archive only replaces an existing grasp if the new one has **HIGHER fitness**.

### 3. **During Replay** (`visualization/replay_grasps.py` + `environments/src/robot_grasping_debug.py`)

```
replay_all_6dof_poses()
  └─> For each grasp in archive:
      ├─> env.reset()
      ├─> env.set_6dof_gripper_pose(gripper_6dof_pose)
      ├─> env.close_gripper()
      │   └─> SAME: wait 200 steps after closing
      │
      └─> apply_all_gripper_shaking_debug(env)  # LINE 133 in replay_grasps.py
          └─> SAME shake logic as training
              └─> Returns: are_all_shakes_successful
```

## Root Cause Analysis

### **The Archive Can Store Non-Deterministic "Lucky" Grasps**

Here's the critical scenario:

1. **Evaluation #1** (iteration 1000):
   - Grasp at position [x=0.018, y=0.039, z=0.007]
   - During shaking: object barely stays in hand (borderline forces ~0.3 N)
   - Physics happens to work out → `is_robust_grasp=True`, fitness=2.0
   - **Stored in archive**

2. **Evaluation #2** (iteration 5000):
   - Different grasp at SAME position [x=0.018, y=0.039, z=0.007] (maps to same cell)
   - This grasp is also robust → `is_robust_grasp=True`, fitness=2.0
   - Archive compares: `candidate_fitness (2.0) > niche_fitness (2.0)` → **FALSE**
   - **Not stored** (keeps the old one from iteration 1000)

3. **Replay**:
   - Replays the grasp from iteration 1000 (the first lucky one)
   - Physics doesn't work out the same way → fails
   - The better grasp from iteration 5000 was never saved!

### **Why Does This Happen?**

#### 1. **Archive Replacement Logic Issue**
```python
if candidate_fitness > niche_fitness:  # Uses > instead of >=
```

When two robust grasps map to the same cell (both fitness=2.0), the archive keeps the **FIRST** one, not necessarily the **BEST** or **MOST STABLE** one.

#### 2. **Fitness is Binary for Robust Grasps**
- Fitness = 0.0 (failed before shaking)
- Fitness = 1.0 (survived one shake)
- Fitness = 2.0 (survived all shakes)

All robust grasps have fitness=2.0, so there's no distinction between:
- A grasp with contact forces of [0.3, 0.4, 0.2] (barely robust)
- A grasp with contact forces of [5.0, 6.0, 7.0] (very robust)

#### 3. **Non-Deterministic Physics**
MuJoCo physics simulations have slight numerical variations:
- Floating-point precision
- Contact solver iterations
- Viewer enabled/disabled timing differences

A grasp with borderline forces (close to 0.1 N threshold) can:
- **Pass during training** (lucky physics iteration)
- **Fail during replay** (slightly different physics outcome)

## Evidence from Terminal Output

```
id=117: marked as robust but are_all_shakes_successful=False
id=85: marked as robust but are_all_shakes_successful=False
```

These were likely borderline grasps that:
1. Got lucky during training evaluation → stored in archive
2. Better grasps evaluated later → rejected because fitness not HIGHER than 2.0
3. During replay → unlucky physics → failed

## Solutions

### **Option 1: Fix Archive Replacement Logic** (Recommended)
Change line 50 in `elite_structured_archive.py`:
```python
if candidate_fitness >= niche_fitness:  # Use >= instead of >
```

This allows newer evaluations of the same fitness to replace older ones, potentially getting more stable grasps.

### **Option 2: Add Robustness Score Beyond Binary Fitness**
Store additional metrics (e.g., minimum contact force) and use them for tie-breaking:
```python
if candidate_fitness > niche_fitness:
    replace = True
elif candidate_fitness == niche_fitness:
    # Tie-breaker: prefer grasp with higher minimum contact force
    candidate_min_force = get_min_contact_force(ind_infos)
    niche_min_force = get_min_contact_force(ind_niche_infos)
    replace = candidate_min_force > niche_min_force
```

### **Option 3: Multiple Replay Trials with Statistics**
Since physics is non-deterministic, test each grasp multiple times:
```python
n_trials = 10
success_count = 0
for trial in range(n_trials):
    if apply_all_gripper_shaking_debug(env):
        success_count += 1
consistency_rate = success_count / n_trials
```

### **Option 4: Increase Force Threshold Margin**
In `env_constants.py`:
```python
CONTACT_FORCE_PARAMETERS = {
    'min_contact_force': 0.2,  # Increase from 0.1 to 0.2
    'min_num_contacts': 3,     # Increase from 2 to 3
}
```

This filters out borderline grasps during training.

## Recommendation

Combine **Option 1** (quick fix) + **Option 4** (preventive):

1. Change `>` to `>=` in archive replacement
2. Increase `min_contact_force` to 0.15-0.2
3. This will:
   - Allow archive to update with newer evaluations of same quality
   - Filter out weak grasps during training
   - Reduce replay failures

## Testing
After implementing fixes, verify consistency rate improves:
```bash
mjpython -m visualization.replay_grasps -r <run_folder> -fs -rb -s
```

Look for: `Consistency rate: X/Y (Z%)` - target should be >90%.

