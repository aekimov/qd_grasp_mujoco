from environments.src.mujoco_simulation.mj_client import MjClient
import environments.src.robots.mj_shadow_hand_consts as sh_consts
import time 
import mujoco

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

client = MjClient(XML_PATH, display=True)


def close_gripper(): #robot_id
    for i_step in range(100):
        actuator_ids, target_positions = client.get_actuators_info(actuator_names=sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS)
        
        for aid, target in zip(actuator_ids, target_positions):
            client.data.ctrl[aid] = target

        client.step()

close_gripper()
# client.close_gripper(actuator_names=sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS)
# client.shake_gripper(animate=True)


while client.viewer and client.viewer.is_running():
    client.viewer.sync()


poses = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]
orient = client.default_gripper_orient


# for p in poses:
#     client.set_6dof_pose_gripper(p, orient)
#     time.sleep(1.0)

# client.reset_gripper_pose()
# client.reset_object_pose()
# time.sleep(1.0)

# client.close_gripper(actuator_names=sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS)
# client.reset_robot_fingers()

# keep the window open (no physics advance)