from environments.src.mujoco_simulation.mj_client import MjClient

XML_PATH = "environments/3d_models/robots/shadow_hand_mujoco/shadow_hand_scene.xml"

client = MjClient(XML_PATH, display=True)

while client.viewer and client.viewer.is_running():
    client.drive_actuator_to_max("rh_A_FFJ3", steps=100, animate=True)

client.close_viewer()


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

