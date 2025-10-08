from environments.src.mujoco_simulation.mj_client import MjClient
import environments.src.robots.mj_shadow_hand_consts as sh_consts
import time 
import mujoco

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

client = MjClient(XML_PATH, display=True)

client.shake_gripper()

while client.viewer and client.viewer.is_running():
    client.viewer.sync()
    time.sleep(0.02)

client.close()