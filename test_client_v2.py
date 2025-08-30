from environments.src.mujoco_simulation.mj_client import MjClient
import environments.src.robots.mj_shadow_hand_consts as sh_consts
import time 
import mujoco

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

client = MjClient(XML_PATH, display=True)

poses = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]
orient = client.default_gripper_orient

# for p in poses:
#     client.set_6dof_pose_gripper(p, orient)
#     if client.viewer:
#         client.viewer.sync()
#     time.sleep(1.0)

# client.reset_gripper_pose()
# client.reset_object_pose()
# if client.viewer:
#     client.viewer.sync()

# close gripper animates only if the function steps AND syncs internally
client.close_gripper(actuator_names=sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS)
client.reset_robot_fingers()
if client.viewer:
    client.viewer.sync()

# keep the window open (no physics advance)
while client.viewer and client.viewer.is_running():
    client.viewer.sync()


