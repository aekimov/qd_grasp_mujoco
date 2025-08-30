import time
import mujoco
import mujoco.viewer

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

# Load model & data
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
adr  = m.jnt_qposadr[jid]   # start index in qpos for this freejoint (7 slots: xyz + quat)
vadr = m.jnt_dofadr[jid]    # start index in qvel (6 slots)

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
    (0.0, 0.0, 0.0),
]

# Optional: settle once before showing the viewer
mujoco.mj_forward(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    for p in poses:
        teleport_hand(p)
        viewer.sync()       # redraw immediately
        time.sleep(1.0)     # just so you can see each teleport

    # keep the viewer open after the loop
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()