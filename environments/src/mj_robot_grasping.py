import numpy as np

from environments.src.mujoco_simulation.mj_client import MjClient
from environments.src.mujoco_simulation.mj_simulation_engine import MjSimulationEngine

import environments.src.env_constants as env_consts
import configs.eval_config as eval_cfg

class MjRobotGrasping:
    def __init__(
            self,
            scene_path,
            list_id_gripper_fingers,
            list_id_gripper_fingers_actuated,
            gripper_6dof_infos,
            gripper_parameters,
            gripper_default_joint_states,
            object_name,
            max_standoff_gripper,
            wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max,
            display=eval_cfg.BULLET_DEFAULT_DISPLAY_FLG,
            debug=False,
            remove_gripper=False,
            **kwargs
            ):

        self._mj_client = MjClient(xml_path=scene_path, display=display)

        self.sim_engine = MjSimulationEngine(
            scene_path=scene_path,
            object_name=object_name,
            mj_client=self._mj_client,
            list_id_gripper_fingers=list_id_gripper_fingers,
            list_id_gripper_fingers_actuated=list_id_gripper_fingers_actuated,
            gripper_6dof_infos=gripper_6dof_infos,
            gripper_parameters=gripper_parameters,
            gripper_default_joint_states=gripper_default_joint_states,
            max_standoff_gripper=max_standoff_gripper,
            wrist_palm_offset_gripper=wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper=half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min=pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max=pose_relative_to_contact_point_d_max
        )

        if remove_gripper:
            raise NotImplementedError('Not implemented remove_gripper function')
        
        self._debug = debug
        self.debug_i_debug_bodies = []

    @property
    def robot_id(self):
        return self.sim_engine.robot_id

    @property
    def obj_id(self):
        return self.sim_engine.obj_id

    @property
    def mj_client(self):
        return self._mj_client

    @property
    def debug(self):
        return self._debug

    @property
    def search_space_bb(self):
        return self.sim_engine.search_space_bb

    @property
    def search_space_bb_object(self):
        return self.sim_engine.search_space_bb_object

    @property
    def gripper_6dof_infos(self):
        return self.sim_engine.gripper_6dof_infos

    @property
    def list_id_gripper_fingers(self):
        return self.sim_engine.list_id_gripper_fingers

    @property
    def list_id_gripper_fingers_actuated(self):
        return self.sim_engine.list_id_gripper_fingers_actuated

    @property
    def max_standoff_gripper(self):
        return self.sim_engine.max_standoff_gripper

    @property
    def wrist_palm_offset_gripper(self):
        return self.sim_engine.wrist_palm_offset_gripper

    @property
    def half_palm_depth_offset_gripper(self):
        return self.sim_engine.half_palm_depth_offset_gripper

    @property
    def pose_relative_to_contact_point_d_min(self):
        return self.sim_engine.pose_relative_to_contact_point_d_min

    @property
    def pose_relative_to_contact_point_d_max(self):
        return self.sim_engine.pose_relative_to_contact_point_d_max

    @property
    def path2obj_point_cloud(self):
        return self.sim_engine.path2obj_point_cloud

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self.sim_engine.list_of_points_for_each_triangle_obj_mesh

    @property
    def object_normals_to_triangles(self):
        return self.sim_engine.object_normals_to_triangles

    @property
    def obj_mesh_vertice_points(self):
        return self.sim_engine.obj_mesh_vertice_points

    def reset(self):
        self.sim_engine.reset(mj_client=self._mj_client)

    def close(self):
        self._mj_client.close()

    def set_6dof_gripper_pose(self, gripper_6dof_pose):
        self.sim_engine.set_6dof_pose_gripper(
            mj_client=self._mj_client,
            start_pos_robot_xyz=gripper_6dof_pose['xyz'],
            start_orient_robot_quat=gripper_6dof_pose['quaternions'],
        )

    def _cvt_genome2synergy_label(self, synergy_label, debug=False):
        raise NotImplementedError('Must be overwritten in robot_grasping subclasses.')

    def _cvt_genome2init_joint_states(self, init_joint_state_genes):
        raise NotImplementedError('Must be overwritten in robot_grasping subclasses.')

    # def close_gripper(self): #robot_id
    #     list_id_grip_fingers_actuated = self.list_id_gripper_fingers_actuated
    #     max_n_step = self.sim_engine.gripper_parameters['max_n_step_close_grip']
    #     is_obj_touched = False

    #     for i_step in range(max_n_step):
    #         actuator_ids, target_positions = self._mj_client.get_actuators_info(actuator_names=list_id_grip_fingers_actuated)
            
    #         for aid, target in zip(actuator_ids, target_positions):
    #             self._mj_client.data.ctrl[aid] = target

    #         self._mj_client.step()

    #     if not is_obj_touched:
    #         is_obj_touched = self.sim_engine.is_grasping_candidate(mj_client=self._mj_client)

    #     return is_obj_touched
    
    def close_gripper(self):
        list_id_grip_fingers_actuated = self.list_id_gripper_fingers_actuated
        max_n_step = self.sim_engine.gripper_parameters['max_n_step_close_grip']

        # Thumb slower than fingers
        speed_factors = {
            "J4": 0.9, "J3": 0.8, "J0": 0.6,           # fingers
            "THJ5": 0.5, "THJ4": 0.4, "THJ3": 0.4,
            "THJ2": 0.4, "THJ1": 0.4,                  # thumb joints slower
        }

        actuator_ids, target_positions = self._mj_client.get_actuators_info(
            actuator_names=list_id_grip_fingers_actuated
        )

        for i_step in range(1, max_n_step + 1):
            progress = i_step / max_n_step
            for aid, name, target in zip(
                actuator_ids, list_id_grip_fingers_actuated, target_positions
            ):
                start = self.sim_engine.gripper_default_joint_states[name]
                factor = 1.0
                for key, val in speed_factors.items():
                    if key in name:
                        factor = val
                        break

                # interpolate: start + factor * progress * (target - start)
                self._mj_client.data.ctrl[aid] = start + factor * progress * (target - start)

            self._mj_client.step()

        # Physics settling: wait for contacts and forces to stabilize
        settling_steps = env_consts.SETTLING_PARAMETERS['after_gripper_close']
        for _ in range(settling_steps):
            self._mj_client.step()

        is_obj_touched = self.sim_engine.is_grasping_candidate(mj_client=self._mj_client)

        return is_obj_touched

    def is_grasping_candidate(self):
        return self.sim_engine.is_grasping_candidate(mj_client=self._mj_client)

    def apply_all_gripper_shaking(self, gripper_6dof_output_data):
        assert not gripper_6dof_output_data['is_overlap']
        assert gripper_6dof_output_data['is_obj_touched']

        are_all_shakes_successful = True
        for idx, i_grip_joint in enumerate(env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']):
            gripper_joint_infos = self.gripper_6dof_infos[i_grip_joint]
            is_being_grasped, n_shake_success = self.apply_gripper_shaking(
                joint_index=i_grip_joint,
                gripper_joint_infos=gripper_joint_infos,
            )
            gripper_6dof_output_data['6dof_data'][i_grip_joint]['is_success'] = is_being_grasped
            at_least_one_failure = are_all_shakes_successful and not is_being_grasped
            if at_least_one_failure:
                are_all_shakes_successful = False
            gripper_6dof_output_data['6dof_data'][i_grip_joint]['n_shake_success'] = n_shake_success
            
            # Add settling delay between different shake axes (but not after the last one)
            if idx < len(env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']) - 1:
                settling_steps = env_consts.SETTLING_PARAMETERS['between_shake_axes']
                for _ in range(settling_steps):
                    self._mj_client.step()

        if are_all_shakes_successful:
            contacts = self._mj_client.get_hand_object_contacts()
            forces = [np.linalg.norm(c['force']) for c in contacts]
            valid_forces = [f for f in forces if f >= env_consts.CONTACT_FORCE_PARAMETERS['min_contact_force']]
            gripper_pos = self._mj_client._get_6dof_pose_gripper()[0]
            print(f"Robust grasp @ [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}] | Contacts: {len(contacts)} | Valid: {len(valid_forces)} | Forces: {[f'{f:.3f}' for f in forces]}")

        gripper_6dof_output_data['is_success'] = True
        gripper_6dof_output_data['is_robust_grasp'] = are_all_shakes_successful
        return gripper_6dof_output_data

    def apply_gripper_shaking(self, joint_index, gripper_joint_infos):
        n_shake = env_consts.SHAKING_PARAMETERS['n_shake']
        target_position = gripper_joint_infos['joint_target_val']
        cmd_jp_kwargs = self.build_cmd_jg_kwargs(
            joint_index=joint_index, gripper_joint_infos=gripper_joint_infos
        )

        is_being_grasped = True
        i_shake = 0

        target_j_poses = [target_position, -target_position, 0]
        while i_shake < n_shake:
            for j_pose in target_j_poses:
                self.command_joint_pose(target_position=j_pose, **cmd_jp_kwargs)
                if not self.sim_engine.is_grasping_candidate(mj_client=self._mj_client):
                    is_being_grasped = False
                    return is_being_grasped, i_shake

            i_shake += 1

        return is_being_grasped, i_shake

    def build_cmd_jg_kwargs(self, joint_index, gripper_joint_infos):
        max_velocity = gripper_joint_infos['max_vel']
        force = gripper_joint_infos['force']
        position_gain = gripper_joint_infos['position_gain']
        velocity_gain = gripper_joint_infos['velocity_gain']
        joint_type = gripper_joint_infos['type']
        joint_axis = gripper_joint_infos['axis']
        return {
            'joint_index': joint_index,
            'max_velocity': max_velocity,
            'force': force,
            'position_gain': position_gain,
            'velocity_gain': velocity_gain,
            'joint_type': joint_type,
            'joint_axis': joint_axis
        }

    def command_joint_pose(
            self,
            joint_index,
            target_position,
            max_velocity,
            force,
            position_gain,
            velocity_gain,
            joint_type,
            joint_axis
    ):
        steps = env_consts.SHAKING_PARAMETERS['t_cmd_stable']
        
        if joint_type == 'prismatic':
            self._mj_client.shake_translation(
                target_position=target_position,
                steps=steps,
                axis=joint_axis
            )
        elif joint_type == 'revolute':
            self._mj_client.shake_rotation(
                target_angle=target_position,
                steps=steps,
                axis=joint_axis
            )
                
    def is_there_overlapping(self):
        return self.sim_engine.is_there_overlapping(mj_client=self._mj_client)

    def set_joint_states_from_genes(self, init_joint_state_genes):

        joint_ids_to_states = self._cvt_genome2init_joint_states(init_joint_state_genes)

        self.sim_engine.set_robot_joint_states(
            mj_client=self._mj_client, joint_ids_to_states=joint_ids_to_states
        )

    def add_noise_to_object_state(self):

        obj_pose_xyz, obj_orient_quat = self.mj_client.getBasePositionAndOrientation(self.obj_id)

        noise2add_obj_pose_xyz = np.random.normal(
            loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_POS_VARIANCE_IN_M, size=3
        )
        noise2add_obj_orient_euler_rpy = np.random.normal(
            loc=0.0, scale=eval_cfg.DOMAIN_RANDOMIZATION_OBJECT_ORIENT_EULER_VARIANCE_IN_RAD, size=3
        )
        noise2add_obj_orient_quat = self.mj_client.getQuaternionFromEuler(noise2add_obj_orient_euler_rpy)

        noisy_obj_pose_xyz = np.array(obj_pose_xyz) + noise2add_obj_pose_xyz
        noisy_obj_orient_quat = np.array(obj_orient_quat) + noise2add_obj_orient_quat

        self.mj_client.resetBasePositionAndOrientation(
            bodyUniqueId=self.obj_id,
            posObj=noisy_obj_pose_xyz,
            ornObj=noisy_obj_orient_quat
        )

    def add_noise_to_friction_coefficients(self):

        noisy_rolling_friction = np.random.uniform(
            low=eval_cfg.DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MIN_VALUE,
            high=eval_cfg.DOMAIN_RANDOMIZATION_ROLLING_FRICTION_MAX_VALUE
        )
        noisy_spinning_friction = np.random.uniform(
            low=eval_cfg.DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MIN_VALUE,
            high=eval_cfg.DOMAIN_RANDOMIZATION_SPINNING_FRICTION_MAX_VALUE
        )

        self.mj_client.changeDynamics(
            bodyUniqueId=self.obj_id, linkIndex=-1,
            rollingFriction=noisy_rolling_friction,
            spinningFriction=noisy_spinning_friction
        )



