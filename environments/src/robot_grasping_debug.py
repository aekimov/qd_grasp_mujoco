import numpy as np
import environments.src.env_constants as env_consts

PATH_TO_DEBUG_SPHERE_GREEN = "environments/3d_models/debug_shape/green_sphere.urdf"
PATH_TO_DEBUG_SPHERE_BLUE = "environments/3d_models/debug_shape/sphere_blue.urdf"
PATH_TO_DEBUG_SPHERE_RED = "environments/3d_models/debug_shape/sphere_red.urdf"


def apply_all_gripper_shaking_debug(robot_grasp_env):
    grip_6dof_infos = robot_grasp_env.sim_engine.gripper_6dof_infos

    are_all_shakes_successful = True
    for idx, i_grip_joint in enumerate(env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']):
        gripper_joint_infos = grip_6dof_infos[i_grip_joint]
        is_being_grasped, n_shake_success = robot_grasp_env.apply_gripper_shaking(
            joint_index=i_grip_joint,
            gripper_joint_infos=gripper_joint_infos,
        )
        at_least_one_failure = are_all_shakes_successful and not is_being_grasped
        if at_least_one_failure:
            are_all_shakes_successful = False
        
        if idx < len(env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']) - 1:
            settling_steps = env_consts.SETTLING_PARAMETERS['between_shake_axes']
            for _ in range(settling_steps):
                robot_grasp_env._mj_client.step()

    if are_all_shakes_successful:
        contacts = robot_grasp_env._mj_client.get_hand_object_contacts()
        forces = [np.linalg.norm(c['force']) for c in contacts]
        valid_forces = [f for f in forces if f >= env_consts.CONTACT_FORCE_PARAMETERS['min_contact_force']]
        gripper_pos = robot_grasp_env._mj_client._get_6dof_pose_gripper()[0]
        print(f"Robust grasp @ [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}] | Contacts: {len(contacts)} | Valid: {len(valid_forces)} | Forces: {[f'{f:.3f}' for f in forces]}")

    return are_all_shakes_successful


