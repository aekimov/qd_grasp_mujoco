from environments.src.mujoco_simulation.mj_client import MjClient

def main():
    XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"
    mj_client = MjClient(XML_PATH, display=True)
    # mj_client.show_aabb()
    mj_client.show_search_space_aabb()

if __name__ == "__main__":
    main()