import time
import numpy as np
import mujoco
import mujoco.viewer
from environments.src.mujoco_simulation.mj_client import MjClient


def main():
    XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"
    mj_client = MjClient(XML_PATH, display=True)
    mj_client.show_debug_cube()

if __name__ == "__main__":
    main()