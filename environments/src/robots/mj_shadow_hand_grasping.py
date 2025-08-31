from pathlib import Path
from environments.src.mj_robot_grasping import MjRobotGrasping
import environments
import environments.src.env_constants as env_consts
import environments.src.robots.mj_shadow_hand_consts as sh_consts

"""
    # ---------------------------------------------------------------------------------------- #
    #                                   SHADOW DEXTEROUS HAND
    # ---------------------------------------------------------------------------------------- #
"""


def init_shadow_hand_scene():
    root_3d_models_robots = \
        Path(environments.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    xml = Path(root_3d_models_robots / sh_consts.SHADOW_HAND_SCENE_RELATIVE_PATH_XML)
    return str(xml)


class MjShadowHand(MjRobotGrasping):

    def __init__(self, **kwargs):
        scene_path = init_shadow_hand_scene()

        list_id_gripper_fingers = sh_consts.ALL_FINGER_ACTUATORS
        list_id_gripper_fingers_actuated = sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS

        gripper_6dof_infos = sh_consts.GRIPPER_6DOF_INFOS
        gripper_parameters = sh_consts.GRIPPER_PARAMETERS

        gripper_default_joint_states = sh_consts.DEFAULT_JOINT_STATES

        max_standoff_gripper = sh_consts.MAX_HAND_STANDOFF
        wrist_palm_offset_gripper = sh_consts.WRIST_PALM_OFFSET
        half_palm_depth_offset_gripper = sh_consts.HALF_PALM_DEPTH

        pose_relative_to_contact_point_d_min = sh_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MIN
        pose_relative_to_contact_point_d_max = sh_consts.POSE_RELATIVE_TO_CONTACT_POINT_D_MAX

        super().__init__(
            scene_path=scene_path,
            list_id_gripper_fingers=list_id_gripper_fingers,
            list_id_gripper_fingers_actuated=list_id_gripper_fingers_actuated,
            gripper_6dof_infos=gripper_6dof_infos,
            gripper_parameters=gripper_parameters,
            gripper_default_joint_states=gripper_default_joint_states,
            max_standoff_gripper=max_standoff_gripper,
            half_palm_depth_offset_gripper=half_palm_depth_offset_gripper,
            wrist_palm_offset_gripper=wrist_palm_offset_gripper,
            pose_relative_to_contact_point_d_min=pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max=pose_relative_to_contact_point_d_max,

            **kwargs,
        )

    def _cvt_genome2synergy_label(self, synergy_label, debug=False):
        return sh_consts.GRIPPER_ACTUATORS_ALL_FINGERS

    def _cvt_genome2init_joint_states(self, init_joint_state_genes):
        raise NotImplementedError('Undefined _cvt_genome2init_joint_states for the current gripper.')
