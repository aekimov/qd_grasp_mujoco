from environments.src.mujoco_simulation.mj_client import MjClient
from environments.src.mj_search_space_bb_processor import get_search_space_bb, get_search_space_bb_side
from environments.src.mujoco_simulation.mj_sim_object import MjSimObject

class MjSimulationEngine:
    def __init__(
            self,
            scene_path,
            object_name,
            mj_client: MjClient,
            list_id_gripper_fingers,
            list_id_gripper_fingers_actuated,
            gripper_6dof_infos,
            gripper_parameters,
            gripper_default_joint_states,
            max_standoff_gripper,
            wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max,
    ):
        self.mj_sim_obj: MjSimObject = None
        self._robot_id = None
        self._search_space_bb = None
        self._search_space_bb_side = None
        self._gripper_6dof_infos = None
        self._gripper_parameters = None
        self._gripper_default_joint_states = None
        self._list_id_gripper_fingers = None
        self._list_id_gripper_fingers_actuated = None
        self._max_standoff_gripper = None
        self._wrist_palm_offset_gripper = None
        self._half_palm_depth_offset_gripper = None
        self._pose_relative_to_contact_point_d_min = None
        self._pose_relative_to_contact_point_d_max = None

        self._init_attributes(
            scene_path=scene_path,
            object_name=object_name,
            mj_client=mj_client,
            list_id_gripper_fingers=list_id_gripper_fingers,
            list_id_gripper_fingers_actuated=list_id_gripper_fingers_actuated,
            gripper_6dof_infos=gripper_6dof_infos,
            gripper_parameters=gripper_parameters,
            gripper_default_joint_states=gripper_default_joint_states,
            max_standoff_gripper=max_standoff_gripper,
            wrist_palm_offset_gripper=wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper=half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min=pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max=pose_relative_to_contact_point_d_max,
        )

    @property
    def obj_id(self):
        return self.mj_sim_obj.obj_id

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def search_space_bb(self):
        return self._search_space_bb

    @property
    def search_space_bb_side(self):
        return self._search_space_bb_side

    @property
    def search_space_bb_object(self):
        return self.mj_sim_obj.search_space_bb_object

    @property
    def gripper_6dof_infos(self):
        return self._gripper_6dof_infos

    @property
    def gripper_default_joint_states(self):
        return self._gripper_default_joint_states

    @property
    def max_standoff_gripper(self):
        return self._max_standoff_gripper

    @property
    def wrist_palm_offset_gripper(self):
        return self._wrist_palm_offset_gripper

    @property
    def half_palm_depth_offset_gripper(self):
        return self._half_palm_depth_offset_gripper

    @property
    def pose_relative_to_contact_point_d_min(self):
        return self._pose_relative_to_contact_point_d_min

    @property
    def pose_relative_to_contact_point_d_max(self):
        return self._pose_relative_to_contact_point_d_max

    @property
    def gripper_parameters(self):
        return self._gripper_parameters

    @property
    def path2obj_point_cloud(self):
        return self.mj_sim_obj.path2obj_point_cloud

    @property
    def list_of_points_for_each_triangle_obj_mesh(self):
        return self.mj_sim_obj.list_of_points_for_each_triangle_obj_mesh

    @property
    def object_normals_to_triangles(self):
        return self.mj_sim_obj.object_normals_to_triangles

    @property
    def obj_mesh_vertice_points(self):
        return self.mj_sim_obj.obj_mesh_vertice_points

    @property
    def list_id_gripper_fingers(self):
        return self._list_id_gripper_fingers

    @property
    def list_id_gripper_fingers_actuated(self):
        return self._list_id_gripper_fingers_actuated

    @property
    def uniform_obj_contact_points(self):
        return self.mj_sim_obj.uniform_obj_contact_points

    @property
    def k_tree_uniform_contact_points(self):
        return self.mj_sim_obj.k_tree_uniform_contact_points

    def _init_attributes(
            self,
            scene_path,
            object_name,
            mj_client: MjClient,
            list_id_gripper_fingers,
            list_id_gripper_fingers_actuated,
            gripper_6dof_infos,
            gripper_parameters,
            gripper_default_joint_states,
            max_standoff_gripper,
            wrist_palm_offset_gripper,
            half_palm_depth_offset_gripper,
            pose_relative_to_contact_point_d_min,
            pose_relative_to_contact_point_d_max,
    ):
        self._gripper_6dof_infos = gripper_6dof_infos
        self._gripper_parameters = gripper_parameters
        self._gripper_default_joint_states = gripper_default_joint_states
        self._list_id_gripper_fingers = list_id_gripper_fingers
        self._list_id_gripper_fingers_actuated = list_id_gripper_fingers_actuated
        self._max_standoff_gripper = max_standoff_gripper
        self._wrist_palm_offset_gripper = wrist_palm_offset_gripper
        self._half_palm_depth_offset_gripper = half_palm_depth_offset_gripper
        self._pose_relative_to_contact_point_d_min = pose_relative_to_contact_point_d_min
        self._pose_relative_to_contact_point_d_max = pose_relative_to_contact_point_d_max

        if max_standoff_gripper is None or wrist_palm_offset_gripper is None or half_palm_depth_offset_gripper is None:
            raise NotImplementedError(
                f'Both max_standoff_gripper and wrist_palm_offset_gripper must be defined for the given gripper.'
            )

        self.mj_sim_obj = MjSimObject(mj_client=mj_client, name="can")
        self._search_space_bb = get_search_space_bb(model=mj_client.model, data=mj_client.data, robot_name="hand_root", object_name="can")
        self._search_space_bb_side = get_search_space_bb_side(self._search_space_bb)
        
        self.reset(mj_client=mj_client)

    def set_6dof_pose_gripper(self, mj_client: MjClient, start_pos_robot_xyz, start_orient_robot_quat):
        mj_client.set_6dof_pose_gripper(start_pos_robot_xyz, start_orient_robot_quat)

    def is_grasping_candidate(self, mj_client: MjClient):
        return mj_client.is_there_contacts()

    def reset(self, mj_client: MjClient):
        mj_client.reset_robot_fingers(default_joint_states=self._gripper_default_joint_states)
        mj_client.reset_object_pose()

    def is_there_overlapping(self, mj_client: MjClient):
        return mj_client.is_there_overlapping()

    def set_robot_joint_states(self, mj_client: MjClient, joint_ids_to_states):
        raise NotImplementedError('Not implemented yet, because it was not called')
