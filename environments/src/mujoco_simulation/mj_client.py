import mujoco
import numpy as np
import time

import environments.src.robots.mj_shadow_hand_consts as sh_consts
import environments.src.env_constants as env_consts
from environments.src.mj_search_space_bb_processor import get_body_aabb, get_search_space_bb, is_descendant_body
from typing import Dict
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC = 0.02 / 5

class MjClient:
    def __init__(self, xml_path: str, display: bool = False):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        self.model = model
        self.data = data
        
        self.default_object_pose, self.default_object_orient = self._get_6dof_pose_object()
        self.default_gripper_pose, self.default_gripper_orient = self._get_6dof_pose_gripper()
        
        self.viewer = None
        
        if display:
            self.open_viewer()
    
    # ----- viewer helpers -----
    def open_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # self.viewer.cam.lookat[2] += 0.5


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
                
    # --- world-pose getters (read-only) ---
    def _get_6dof_pose_gripper(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        bid = self.model.jnt_bodyid[jid]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def _get_6dof_pose_object(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        bid = self.model.jnt_bodyid[jid]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    # --- setters (write joint state; quat is wxyz) ---
    def _set_6dof_pose_gripper_joint(self, pos_xyz, quat_wxyz):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        adr = self.model.jnt_qposadr[jid]   # start index in qpos for this freejoint (7 slots: xyz + quat)
        vadr = self.model.jnt_dofadr[jid]    # start index in qvel (6 slots)
        
        self.data.qpos[adr:adr+3] = pos_xyz
        self.data.qpos[adr+3:adr+7] = quat_wxyz
        self.data.qvel[vadr:vadr+6] = 0  # Zero base velocities so we don't inject impulses
        
        self.forward()
        
    # added weld constraint to and now both mocap and freejoint are needed to be moves
    # so the hand base does not drift/rotate during grasp
    def set_6dof_pose_gripper(self, pos_xyz, quat_wxyz):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
        mid = self.model.body_mocapid[bid]
        
        jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        qadr = self.model.jnt_qposadr[jid]
        vadr = self.model.jnt_dofadr[jid]

        self.data.mocap_pos[mid] = pos_xyz
        self.data.mocap_quat[mid] = quat_wxyz
        self.data.qpos[qadr:qadr+3] = pos_xyz
        self.data.qpos[qadr+3:qadr+7] = quat_wxyz
        # self.data.qvel[vadr:vadr+6] = 0.0
        
        self.forward()

    def set_6dof_pose_gripper_shake(self, pos_xyz, quat_wxyz):
        """Only set mocap target - let weld constraint handle the rest"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
        mid = self.model.body_mocapid[bid]
        
        self.data.mocap_pos[mid] = pos_xyz
        self.data.mocap_quat[mid] = quat_wxyz
        
        self.forward()
    
    def set_6dof_pose_object(self, pos_xyz, quat_wxyz):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        qadr = self.model.jnt_qposadr[jid]
        vadr = self.model.jnt_dofadr[jid]
        
        self.data.qpos[qadr:qadr+3] = pos_xyz
        self.data.qpos[qadr+3:qadr+7] = quat_wxyz
        self.data.qvel[vadr:vadr+6] = 0
        
        self.forward()
    
    def forward(self):
        mujoco.mj_forward(self.model, self.data)
        
        if self.viewer:
            self.viewer.sync()
            
    def reset_gripper_pose(self):
        self.set_6dof_pose_gripper(self.default_gripper_pose, self.default_gripper_orient)
        
    def reset_object_pose(self):
        self.set_6dof_pose_object(self.default_object_pose, self.default_object_orient)
            
    def reset_robot_fingers(self, default_joint_states: Dict[str, float]):
        for actuator_name, default_val in default_joint_states.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            if actuator_name in sh_consts.TENDON_TO_JOINTS:
                    joint_names = sh_consts.TENDON_TO_JOINTS[actuator_name]
                    share = float(default_val) / len(joint_names)
                    for jname in joint_names:
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        qadr = self.model.jnt_qposadr[jid]
                        dadr = self.model.jnt_dofadr[jid]
                        self.data.qpos[qadr] = share
                        self.data.qvel[dadr] = 0.0
                    # Set actuator control to maintain this position
                    self.data.ctrl[aid] = float(default_val)
            else:
                target_id = int(self.model.actuator_trnid[aid, 0])
                qadr = self.model.jnt_qposadr[target_id]
                dadr = self.model.jnt_dofadr[target_id]
                self.data.qpos[qadr] = float(default_val)
                self.data.qvel[dadr] = 0.0
                # Set actuator control to maintain this position
                self.data.ctrl[aid] = float(default_val)

        self.forward()
             
    def reset(self):
        self.reset_gripper_pose()
        self.reset_object_pose()
        # self.reset_robot_fingers()
            
    def get_actuators_info(self, actuator_names: list[str]):
        actuator_ids = []
        target_positions = []
        
        for name in actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            target = self.model.actuator_ctrlrange[aid][1]
            actuator_ids.append(aid)
            target_positions.append(float(target))

        return actuator_ids, target_positions
                
    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)
            else:
                print("\nViewer was closed by user. Terminating simulation...")
                self.viewer = None
                raise KeyboardInterrupt("Viewer was closed by user")

    def shake_translation(self, target_position, steps, axis='x'):
        """Apply translation shake along specified axis"""
        p0, q0 = self._get_6dof_pose_gripper()
        
        axis_vector = np.zeros(3)
        if axis == 'x':
            axis_vector[0] = 1.0
        elif axis == 'y':
            axis_vector[1] = 1.0
        elif axis == 'z':
            axis_vector[2] = 1.0
        
        p_goal = p0 + target_position * axis_vector
        p_start = self.data.mocap_pos[self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
        ]].copy()
        
        for k in range(steps):
            t = (k + 1) / float(steps)
            p = (1.0 - t) * p_start + t * p_goal
            self.set_6dof_pose_gripper_shake(p, q0)
            self.step()
    
    def shake_rotation(self, target_angle, steps, axis='y'):
        hand_offset = np.array([0.33, 0.0, 0.02])
        mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")]
        
        p_current = self.data.mocap_pos[mocap_id].copy()
        q_current = self.data.mocap_quat[mocap_id].copy()
        q_scipy = q_current[[1, 2, 3, 0]]
        
        rot_current = R.from_quat(q_scipy)
        rot_goal = R.from_euler(axis, target_angle, degrees=False) * rot_current
        slerp = Slerp([0, 1], R.from_quat([q_scipy, rot_goal.as_quat()]))
        pivot_point = p_current - rot_current.apply(hand_offset) + hand_offset
        
        for k in range(steps):
            rot_interp = slerp([(k + 1) / steps])[0]
            p_interp = pivot_point + rot_interp.apply(hand_offset) - hand_offset
            q_interp = rot_interp.as_quat()[[3, 0, 1, 2]]
            self.set_6dof_pose_gripper_shake(p_interp, q_interp)
            self.step()
        
    # Not used directly, just in the shake test
    def shake_gripper(self):
        """Test function for visualizing both translation and rotation shakes"""
        translation_amp = 0.2
        rotation_amp = np.pi / 4
        n_shake = env_consts.SHAKING_PARAMETERS['n_shake']
        hold = env_consts.SHAKING_PARAMETERS.get('t_cmd_stable', 50)
        targets = [1.0, -1.0, 0.0]
        
        print(f"\nShake test: translation={translation_amp}m, rotation={np.degrees(rotation_amp):.1f}°, n_shake={n_shake}")
        
        for i_shake in range(n_shake):            
            for dx_factor in targets:
                self.shake_translation(translation_amp * dx_factor, hold, axis='x')
            
            for angle_factor in targets:
                self.shake_rotation(rotation_amp * angle_factor, hold, axis='y')

        self.reset()
        
    def is_there_contacts(self) -> bool:
        contacts = self.get_hand_object_contacts()
        return len(contacts) != 0
    
    # def are_bodies_in_contact(self, body_names: list[str], target_body_name: str) -> bool:
    #     m, d = self.model, self.data

    #     target_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    #     body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in body_names]

    #     for i in range(d.ncon):
    #         c = d.contact[i]
    #         g1, g2 = int(c.geom1), int(c.geom2)
    #         b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])

    #         if ((b1 == target_bid and b2 in body_ids) or
    #             (b2 == target_bid and b1 in body_ids)):
    #             return True
    #     return False
    
    def is_there_overlapping(self) -> bool:
        """Check for penetrating contacts involving the robot hand"""
        contacts = self.get_hand_object_contacts()
        
        return len(contacts) != 0
        # Check if any contact has negative distance (penetration)
        for contact in contacts:
            if contact['dist'] < 0.0:  # penetration
                return True
        return False

    def draw_line(self, pos1, pos2, width=3.0, rgba=(1, 0, 0, 1)):
        scn = self.viewer.user_scn
        g = scn.geoms[scn.ngeom]
        p0 = np.asarray(pos1, float)
        p1 = np.asarray(pos2, float)
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_LINE, float(width),
            p0, p1
        )
        g.rgba[:] = rgba
        scn.ngeom += 1

    def show_debug_cube(self, center=(0,0,0.2), half_size=(0.05,0.05,0.05), rgba=(0,1,0,0.5)):
        while self.viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            scn = self.viewer.user_scn
            scn.ngeom = 0
            g = scn.geoms[scn.ngeom]
            
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=half_size,
                pos=center,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )
            scn.ngeom += 1
            self.viewer.sync()
            
    def get_hand_object_contacts(self, robot_name="hand_root", object_name="can"):
        robot_body_id = self.model.body(robot_name).id
        object_body_id = self.model.body(object_name).id
        
        contacts = []
        
        # Loop through all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get body IDs for both geoms in contact
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            
            # Check if one geom belongs to robot subtree and other to object
            robot_geom1 = is_descendant_body(self.model, robot_body_id, body1)
            robot_geom2 = is_descendant_body(self.model, robot_body_id, body2)
            object_geom1 = is_descendant_body(self.model, object_body_id, body1)
            object_geom2 = is_descendant_body(self.model, object_body_id, body2)
            
            if (robot_geom1 and object_geom2) or (robot_geom2 and object_geom1):
                # This is a robot-object contact
                # Extract 6D force (3 force + 3 torque) in contact frame
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                
                contacts.append({
                    'pos': contact.pos.copy(),           # contact position
                    'dist': contact.dist,                 # penetration depth (negative)
                    'normal': contact.frame[:3].copy(),   # contact normal
                    'force': force[:3].copy(),            # force vector
                    'torque': force[3:].copy(),           # torque vector
                    'geom1': contact.geom1,               # geom IDs
                    'geom2': contact.geom2
                })
        
        return contacts

    def show_aabb(self, robot_name="hand_root", object_name="can"):
        print(f"Getting AABB for robot: {robot_name}")
        robot_aabb_min, robot_aabb_max = get_body_aabb(self.model, self.data, robot_name)
        print(f"Robot AABB: min={robot_aabb_min}, max={robot_aabb_max}")
        
        print(f"Getting AABB for object: {object_name}")  
        object_aabb_min, object_aabb_max = get_body_aabb(self.model, self.data, object_name)
        print(f"Object AABB: min={object_aabb_min}, max={object_aabb_max}")
        
        robot_center = (robot_aabb_min + robot_aabb_max) / 2
        robot_half_sizes = (robot_aabb_max - robot_aabb_min) / 2
        
        object_center = (object_aabb_min + object_aabb_max) / 2
        object_half_sizes = (object_aabb_max - object_aabb_min) / 2
        
        print(f"Robot - Center: {robot_center}, Half sizes: {robot_half_sizes}")
        print(f"Object - Center: {object_center}, Half sizes: {object_half_sizes}")
        
        while self.viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            scn = self.viewer.user_scn
            scn.ngeom = 0
            
            # Robot AABB (red)
            g1 = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g1,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=robot_half_sizes,
                pos=robot_center,
                mat=np.eye(3).flatten(),
                rgba=(1, 0, 0, 0.3)
            )
            scn.ngeom += 1
            
            # Object AABB (blue)
            g2 = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g2,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=object_half_sizes,
                pos=object_center,
                mat=np.eye(3).flatten(),
                rgba=(0, 0, 1, 0.3)
            )
            scn.ngeom += 1
            self.viewer.sync()

    def show_search_space_aabb(self, robot_name="hand_root", object_name="can"):
        print(f"Getting search space AABB for robot '{robot_name}' and object '{object_name}'")
        ss_bb = get_search_space_bb(self.model, self.data, robot_name, object_name)
        print(f"Search space BB: min={ss_bb.aabb_min}, max={ss_bb.aabb_max}")
        
        center = (np.array(ss_bb.aabb_min) + np.array(ss_bb.aabb_max)) / 2
        half_sizes = (np.array(ss_bb.aabb_max) - np.array(ss_bb.aabb_min)) / 2
        print(f"Search space - Center: {center}, Half sizes: {half_sizes}")
        
        while self.viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            scn = self.viewer.user_scn
            scn.ngeom = 0
            
            # Search space AABB (green)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=half_sizes,
                pos=center,
                mat=np.eye(3).flatten(),
                rgba=(0, 1, 0, 0.3)
            )
            scn.ngeom += 1
            self.viewer.sync()
            
            
    def debug_clear(self):
        self.viewer.user_scn.ngeom = 0

    def debug_sync(self):
        self.viewer.sync()

    def draw_triangle(self, tri, width=5.0, rgba=(1, 0, 0, 1)):
        scn = self.viewer.user_scn
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_LINE, float(width),
                np.asarray(a, float), np.asarray(b, float)
            )
            g.rgba[:] = rgba
            scn.ngeom += 1

    def draw_sphere(self, pos, radius=0.01, rgba=(1, 0, 1, 1)):
        scn = self.viewer.user_scn
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(radius, radius, radius),
            pos=np.asarray(pos, float),
            mat=(1,0,0, 0,1,0, 0,0,1),
            rgba=rgba,
        )
        scn.ngeom += 1
        
    def draw_normal(self, pos, normal, scale=0.2, width=3.0, rgba=(0, 1, 0, 1)):
        """
        Draw a line showing the normal direction starting at pos.
        """
        scn = self.viewer.user_scn
        g = scn.geoms[scn.ngeom]
        p0 = np.asarray(pos, float)
        p1 = p0 + scale * np.asarray(normal, float)
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_LINE, float(width),
            p0, p1
        )
        g.rgba[:] = rgba
        scn.ngeom += 1
        
    def draw_box(self, pos, half_extents, rgba=(1, 0, 0, 1)):
        """Draw a box at the specified position with given half extents and color."""
        scn = self.viewer.user_scn
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_extents,
            pos=np.asarray(pos, float),
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        scn.ngeom += 1
        
        
    def draw_aabb(self, robot_name="hand_root", object_name="can"):
        """Draw AABB boxes for robot (red) and object (blue) - single frame, no loop"""
        robot_aabb_min, robot_aabb_max = get_body_aabb(self.model, self.data, robot_name)
        object_aabb_min, object_aabb_max = get_body_aabb(self.model, self.data, object_name)
        
        robot_center = (robot_aabb_min + robot_aabb_max) / 2
        robot_half_sizes = (robot_aabb_max - robot_aabb_min) / 2
        
        object_center = (object_aabb_min + object_aabb_max) / 2
        object_half_sizes = (object_aabb_max - object_aabb_min) / 2
        
        # Draw robot AABB (red, transparent)
        self.draw_box(pos=robot_center, half_extents=robot_half_sizes, rgba=(1, 0, 0, 0.3))
        
        # Draw object AABB (blue, transparent)
        self.draw_box(pos=object_center, half_extents=object_half_sizes, rgba=(0, 0, 1, 0.3))

    def draw_search_space_aabb(self, robot_name="hand_root", object_name="can"):
        """Draw search space AABB (green) - single frame, no loop"""
        ss_bb = get_search_space_bb(self.model, self.data, robot_name, object_name)
        
        center = (np.array(ss_bb.aabb_min) + np.array(ss_bb.aabb_max)) / 2
        half_sizes = (np.array(ss_bb.aabb_max) - np.array(ss_bb.aabb_min)) / 2
        
        # Draw search space AABB (green, transparent)
        self.draw_box(pos=center, half_extents=half_sizes, rgba=(0, 1, 0, 0.3))