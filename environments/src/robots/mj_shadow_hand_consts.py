"""
MuJoCo Shadow Hand Actuator Constants
All actuator names correspond to the position actuators defined in right_hand.xml.
"""
import numpy as np

GRIPPER_6DOF_INFOS = {
    0: {'axis': 'x', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2}, # rouge
    1: {'axis': 'y', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82, 'velocity_gain': 64, 'joint_target_val': 0.2},  # vert
    2: {'axis': 'z', 'type': 'prismatic', 'max_vel': 10, 'force': 20, 'position_gain': 2.82,  'velocity_gain': 64, 'joint_target_val': 0.2},  # bleu

    3: {'axis': 'x', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    4: {'axis': 'y', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
    5: {'axis': 'z', 'type': 'revolute', 'max_vel': 10, 'force': 100, 'position_gain': 0.05,  'velocity_gain': 1, 'joint_target_val': np.pi/4},
}


# ========================================
# WRIST ACTUATORS
# ========================================

# Wrist actuators (6DOF hand positioning)
A_WRJ2 = "rh_A_WRJ2"  # Wrist Y rotation (forearm to wrist)
A_WRJ1 = "rh_A_WRJ1"  # Wrist X rotation (wrist to palm)

WRIST_ACTUATORS = [A_WRJ2, A_WRJ1]


# ========================================
# FINGER ACTUATORS
# ========================================

# Index Finger (FF = First Finger)
A_FFJ4 = "rh_A_FFJ4"  # Knuckle (abduction/adduction)
A_FFJ3 = "rh_A_FFJ3"  # Proximal joint
A_FFJ0 = "rh_A_FFJ0"  # Middle+Distal joints (coupled via tendon)
 
INDEX_FINGER_ACTUATORS = [A_FFJ4, A_FFJ3, A_FFJ0]

# Middle Finger (MF = Middle Finger) 
A_MFJ4 = "rh_A_MFJ4"  # Knuckle (abduction/adduction)
A_MFJ3 = "rh_A_MFJ3"  # Proximal joint
A_MFJ0 = "rh_A_MFJ0"  # Middle+Distal joints (coupled via tendon)

MIDDLE_FINGER_ACTUATORS = [A_MFJ4, A_MFJ3, A_MFJ0]

# Ring Finger (RF = Ring Finger)
A_RFJ4 = "rh_A_RFJ4"  # Knuckle (abduction/adduction)
A_RFJ3 = "rh_A_RFJ3"  # Proximal joint
A_RFJ0 = "rh_A_RFJ0"  # Middle+Distal joints (coupled via tendon)

RING_FINGER_ACTUATORS = [A_RFJ4, A_RFJ3, A_RFJ0]

# Little Finger (LF = Little Finger)
A_LFJ5 = "rh_A_LFJ5"  # Metacarpal (finger base rotation)
A_LFJ4 = "rh_A_LFJ4"  # Knuckle (abduction/adduction)
A_LFJ3 = "rh_A_LFJ3"  # Proximal joint
A_LFJ0 = "rh_A_LFJ0"  # Middle+Distal joints (coupled via tendon)

LITTLE_FINGER_ACTUATORS = [A_LFJ5, A_LFJ4, A_LFJ3, A_LFJ0]

# Thumb (TH = Thumb)
A_THJ5 = "rh_A_THJ5"  # Base rotation
A_THJ4 = "rh_A_THJ4"  # Proximal joint
A_THJ3 = "rh_A_THJ3"  # Hub joint
A_THJ2 = "rh_A_THJ2"  # Middle joint
A_THJ1 = "rh_A_THJ1"  # Distal joint

THUMB_ACTUATORS = [A_THJ5, A_THJ4, A_THJ3, A_THJ2, A_THJ1]


# ========================================
# GRIPPER CLOSING ACTUATOR GROUPS
# ========================================

# Primary actuators for finger closing (most important for grasping)
# These correspond to the main finger joints that close toward the palm

# Index finger closing actuators
GRIPPER_ACTUATORS_INDEX = [A_FFJ3, A_FFJ0]  # Proximal + Middle/Distal

# Middle finger closing actuators  
GRIPPER_ACTUATORS_MIDDLE = [A_MFJ3, A_MFJ0]  # Proximal + Middle/Distal

# Ring finger closing actuators
GRIPPER_ACTUATORS_RING = [A_RFJ3, A_RFJ0]  # Proximal + Middle/Distal

# Little finger closing actuators
GRIPPER_ACTUATORS_LITTLE = [A_LFJ3, A_LFJ0]  # Proximal + Middle/Distal

# Thumb closing actuators (for opposition grasp)
GRIPPER_ACTUATORS_THUMB_OPPOSITION = [A_THJ4, A_THJ3, A_THJ2, A_THJ1]

# Thumb closing actuators (for palm grasp - includes base)
GRIPPER_ACTUATORS_THUMB_PALM = [A_THJ5, A_THJ4, A_THJ3, A_THJ2, A_THJ1]

# Default thumb actuators for general grasping
GRIPPER_ACTUATORS_THUMB = GRIPPER_ACTUATORS_THUMB_PALM


# ========================================
# COMPLETE GRIPPER ACTUATOR LISTS
# ========================================

# All primary finger closing actuators (most commonly used)
GRIPPER_ACTUATORS_ALL_FINGERS = (GRIPPER_ACTUATORS_INDEX + 
                                 GRIPPER_ACTUATORS_MIDDLE + 
                                 GRIPPER_ACTUATORS_RING + 
                                 GRIPPER_ACTUATORS_LITTLE + 
                                 GRIPPER_ACTUATORS_THUMB)

# All actuators including knuckle/abduction controls
ALL_FINGER_ACTUATORS = (INDEX_FINGER_ACTUATORS + 
                       MIDDLE_FINGER_ACTUATORS + 
                       RING_FINGER_ACTUATORS + 
                       LITTLE_FINGER_ACTUATORS + 
                       THUMB_ACTUATORS)

# Complete actuator list including wrist
ALL_ACTUATORS = WRIST_ACTUATORS + ALL_FINGER_ACTUATORS


# ========================================
# GRIPPER PARAMETERS
# ========================================

GRIPPER_PARAMETERS = {
    'max_n_step_close_grip': 100,
    'max_velocity_maintain_6dof': 10,
    'force_maintain_6dof': 1000
}


# ========================================
# DEFAULT TARGET POSITIONS
# ========================================

DEFAULT_JOINT_STATES = {
    A_WRJ2: 0.0,
    A_WRJ1: 0.0,
    
    A_FFJ4: 0.0,
    A_FFJ3: 0.0,
    A_FFJ0: 0.0,
    
    A_MFJ4: 0.0,
    A_MFJ3: 0.0,
    A_MFJ0: 0.0,
    
    A_RFJ4: 0.0,
    A_RFJ3: 0.0,
    A_RFJ0: 0.0,
        
    A_LFJ5: 0.0,
    A_LFJ4: 0.0,
    A_LFJ3: 0.0,
    A_LFJ0: 0.0,
    
    A_THJ5: 0.0,
    A_THJ4: 1.22,
    A_THJ3: 0.0,
    A_THJ2: 0.0,
    A_THJ1: 0.0,
}

TENDON_TO_JOINTS = {
    A_FFJ0: ("rh_FFJ2", "rh_FFJ1"),
    A_MFJ0: ("rh_MFJ2", "rh_MFJ1"),
    A_RFJ0: ("rh_RFJ2", "rh_RFJ1"),
    A_LFJ0: ("rh_LFJ2", "rh_LFJ1")
}


# ---------------------------------------------- #
#                   KEY MEASURES
# ---------------------------------------------- #


WRIST_PALM_OFFSET = 0.0  # CODE DE MATHILDE : "PALM_WIRST"

# EXTREMITY_PALM represents the distance from the palm base to the fingertip extremities when the hand is fully extended.
EXTREMITY_PALM = 0.145
HALF_PALM_DEPTH = 0.001

MAX_HAND_STANDOFF = EXTREMITY_PALM

POSE_RELATIVE_TO_CONTACT_POINT_D_MIN = WRIST_PALM_OFFSET + HALF_PALM_DEPTH
POSE_RELATIVE_TO_CONTACT_POINT_D_MAX = POSE_RELATIVE_TO_CONTACT_POINT_D_MIN + MAX_HAND_STANDOFF


# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

SHADOW_HAND_SCENE_RELATIVE_PATH_XML = "shadow_hand_mujoco/shadow_hand_scene.xml"