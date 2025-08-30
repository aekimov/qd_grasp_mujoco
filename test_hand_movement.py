import time
import mujoco
import mujoco.viewer

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

# Load model & data
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d) # Optional: settle once before showing the viewer

jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
adr  = m.jnt_qposadr[jid]   # start index in qpos for this freejoint (7 slots: xyz + quat)
vadr = m.jnt_dofadr[jid]    # start index in qvel (6 slots)

DEFAULT_ROBOT_POSE   = d.qpos[adr:adr+3].copy()
DEFAULT_ROBOT_ORIENT = d.qpos[adr+3:adr+7].copy()

print(f"default_robot_pose: {DEFAULT_ROBOT_POSE}, default_robot_orient: {DEFAULT_ROBOT_ORIENT}")
# bid_can = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "can")
# pos_world, quat_world = d.xpos[bid_can].copy(), d.xquat[bid_can].copy()

def teleport_hand(p):  # p = (x, y, z) in meters, world frame
    # Set pose: position then identity quaternion (no rotation)
    d.qpos[adr:adr+3] = p
    d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    # Zero base velocities so we don't inject impulses
    d.qvel[vadr:vadr+6] = 0
    # Recompute derived quantities (contacts, kinematics, etc.)
    mujoco.mj_forward(m, d)

poses = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    DEFAULT_ROBOT_POSE
]

with mujoco.viewer.launch_passive(m, d) as viewer:
    for p in poses:
        teleport_hand(p)
        viewer.sync()       # redraw immediately
        time.sleep(1.0)     # just so you can see each teleport

    # keep the viewer open after the loop
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()