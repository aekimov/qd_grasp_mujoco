
import numpy as np

from utils.common_tools import project_from_to_value

from algorithms.evaluation.grasp_strategies_routines import get_normal_surface_point, get_gripper_pose_relatively_to_contact_point_allegro_compatible

import configs.qd_config as qd_cfg

def cvt_genome_to_preset_6dof_pose_contact_strat_contact_point_finder(robot_grasp_env, genome):

    # Extract values from the genome
    contact_point_finder_xyz_pose_genome_values = genome[:3]
    nu_genome = genome[3]
    d_genome = genome[4]
    ksi_genome = genome[5]
    omega_genome = genome[6]

    # Get contact point finder voxel 3D pose
    contact_point_finder_xyz_pose = [
        project_from_to_value(
            interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
            interval_to=[ss_min, ss_max],
            x_start=gen_val
        )
        for gen_val, ss_min, ss_max in zip(
            contact_point_finder_xyz_pose_genome_values,
            robot_grasp_env.search_space_bb_object.aabb_min,
            robot_grasp_env.search_space_bb_object.aabb_max
        )
    ]

    # Compute the gripper position and orientation from the contact point
    nu_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, np.pi],
        x_start=nu_genome
    )

    # nu_projected = np.pi

    d_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[robot_grasp_env.pose_relative_to_contact_point_d_min, robot_grasp_env.pose_relative_to_contact_point_d_max],
        x_start=d_genome
    )

    ksi_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=ksi_genome
    )
    omega_projected = project_from_to_value(
        interval_from=qd_cfg.FIXED_INTERVAL_GENOME,
        interval_to=[0, 2 * np.pi],
        x_start=omega_genome
    )

    hand_pose_from_contact_params = {
        'nu': nu_projected,
        'd': d_projected,
        'ksi': ksi_projected,
        'omega': omega_projected,
    }

    return contact_point_finder_xyz_pose, hand_pose_from_contact_params

def cvt_genome_to_6dof_pose_contact_strategy_search(robot_grasp_env, genome, robot):

    contact_point_finder_xyz_pose, hand_pose_from_contact_params = \
        cvt_genome_to_preset_6dof_pose_contact_strat_contact_point_finder(robot_grasp_env=robot_grasp_env, genome=genome)

    # Get closer point on the object surface
    query = [contact_point_finder_xyz_pose]
    closest_contact_point_id = robot_grasp_env.sim_engine.k_tree_uniform_contact_points.kneighbors(X=query)[1][0][0]
    closest_contact_point = robot_grasp_env.sim_engine.uniform_obj_contact_points[closest_contact_point_id]
    
    debug = robot_grasp_env.debug
    # print(robot_grasp_env) #environments.src.robots.mj_shadow_hand_grasping.MjShadowHand
    
    # Apply standard approach-based method
    normal_at_contact_point = get_normal_surface_point(
        bullet_client=robot_grasp_env.mj_client,
        list_of_points_for_each_triangle_object_mesh=robot_grasp_env.list_of_points_for_each_triangle_obj_mesh,
        object_normals_to_triangles=robot_grasp_env.object_normals_to_triangles,
        contact_point=closest_contact_point,
        debug=debug
    )

    gripper_6dof_pose = get_gripper_pose_relatively_to_contact_point_allegro_compatible(
        bullet_client=robot_grasp_env.mj_client,
        hand_pose_from_contact_params=hand_pose_from_contact_params,
        normal_at_contact_point=normal_at_contact_point,
        contact_point=closest_contact_point,
        wrist_palm_offset_gripper=robot_grasp_env.wrist_palm_offset_gripper,
        debug=debug,
        robot=robot,
    )

    if debug:
        hand_pos = gripper_6dof_pose['xyz']
        hand_quat = gripper_6dof_pose['quaternions']
        distance_to_contact = np.linalg.norm(np.array(hand_pos) - np.array(closest_contact_point))
        standoff_d = hand_pose_from_contact_params['d']
        
        print(f"Contact: [{closest_contact_point[0]:.3f}, {closest_contact_point[1]:.3f}, {closest_contact_point[2]:.3f}]")
        print(f"Hand XYZ: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        print(f"Hand quat (WXYZ): [{hand_quat[0]:.3f}, {hand_quat[1]:.3f}, {hand_quat[2]:.3f}, {hand_quat[3]:.3f}]")
        print(f"Standoff d: {standoff_d:.3f}, Actual distance: {distance_to_contact:.3f}")
        print(f"Angles - nu: {hand_pose_from_contact_params['nu']:.3f} rad ({np.degrees(hand_pose_from_contact_params['nu']):.1f}°), ksi: {hand_pose_from_contact_params['ksi']:.3f} rad ({np.degrees(hand_pose_from_contact_params['ksi']):.1f}°), omega: {hand_pose_from_contact_params['omega']:.3f} rad ({np.degrees(hand_pose_from_contact_params['omega']):.1f}°)")
        print("=" * 60)
        
        robot_grasp_env._mj_client.draw_line(closest_contact_point, hand_pos, width=2.0, rgba=(0, 1, 0, 1))
        robot_grasp_env._mj_client.debug_sync()
    return gripper_6dof_pose