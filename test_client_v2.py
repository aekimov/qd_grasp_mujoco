from environments.src.mujoco_simulation.mj_client import MjClient
import environments.src.robots.mj_shadow_hand_consts as sh_consts
import time 

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

client = MjClient(XML_PATH, display=True)

# client.close_gripper("rh_A_FFJ3")
client.reset_robot_fingers()
client.close_gripper(actuator_names=sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS)

print("Resetting robot fingers")
client.reset_robot_fingers()

while client.viewer and client.viewer.is_running():
    client.viewer.sync()     # redraw same frame
    # time.sleep(1/60)       # optional pacing
    

# while client.viewer and client.viewer.is_running():
#    client.step(1, sync=True)

# client.close_viewer()


# client.servo_ramp("rh_A_FFJ3", secs=5.0)

# while client.viewer and client.viewer.is_running():
#     client.step(1, sync=True)


# poses = [
#     (1.0, 0.0, 0.0),
#     (0.0, 1.0, 0.0),
#     (0.0, 0.0, 1.0)
# ]

# orient = client.default_gripper_orient

# with mujoco.viewer.launch_passive(client.model, client.data) as viewer:
#     for p in poses:
#         client.set_6dof_pose_gripper(p, orient)
#         viewer.sync()       # redraw immediately
#         time.sleep(1.0)     # just so you can see each teleport

#     client.reset_gripper_pose()
#     client.reset_object_pose()
    
#     print(client.get_actuator_info("rh_A_FFJ3"))
#     client.drive_actuator_to_max("rh_A_FFJ3")

#     # keep the viewer open after the loop
#     while viewer.is_running():
#         mujoco.mj_step(client.model, client.data)
#         viewer.sync()

