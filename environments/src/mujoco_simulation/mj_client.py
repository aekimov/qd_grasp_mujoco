import mujoco
import numpy as np
from typing import Tuple
import time

import environments.src.robots.mj_shadow_hand_consts as sh_consts
import environments.src.env_constants as env_consts

MAX_STEP_CLOSE_GRIP = 100
TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC = 0.02 * 2

class MjClient:
    def __init__(self, xml_path: str, display: bool = False):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # settle derived state once
        
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
            self.viewer.cam.lookat[2] += 0.5


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
                
    # --- world-pose getters (read-only) ---
    def _get_6dof_pose_gripper(self) -> Tuple[np.ndarray, np.ndarray]:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        bid = self.model.jnt_bodyid[jid]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def _get_6dof_pose_object(self) -> Tuple[np.ndarray, np.ndarray]:
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
        # self.data.qpos[qadr:qadr+3] = pos_xyz
        # self.data.qpos[qadr+3:qadr+7] = quat_wxyz
        # self.data.qvel[vadr:vadr+6] = 0.0
        
        # self.forward()

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
            
    def reset_robot_fingers(self):
        for actuator_name, default_val in sh_consts.DEFAULT_JOINT_STATES.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            if actuator_name in sh_consts.TENDON_TO_JOINTS:
                    joint_names = sh_consts.TENDON_TO_JOINTS[actuator_name]
                    share = float(default_val) / len(joint_names) # split the desired tendon target evenly across its joints
                    for jname in joint_names:
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        qadr = self.model.jnt_qposadr[jid]
                        dadr = self.model.jnt_dofadr[jid]
                        self.data.qpos[qadr] = share
                        self.data.qvel[dadr] = 0.0
            else:
                target_id = int(self.model.actuator_trnid[aid, 0])
                qadr = self.model.jnt_qposadr[target_id]
                dadr = self.model.jnt_dofadr[target_id]
                self.data.qpos[qadr] = float(default_val)
                self.data.qvel[dadr] = 0.0

        self.forward()
             
    def reset(self):
        self.reset_gripper_pose()
        self.reset_object_pose()
        self.reset_robot_fingers()
    
    def close_gripper(self, actuator_names: list[str]):
        actuator_ids = []
        target_positions = []
        
        for name in actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            target = self.model.actuator_ctrlrange[aid][1]
            actuator_ids.append(aid)
            target_positions.append(float(target))

        for _ in range(MAX_STEP_CLOSE_GRIP):
            for aid, target in zip(actuator_ids, target_positions):
                self.data.ctrl[aid] = target

            mujoco.mj_step(self.model, self.data)

            if self.viewer:
                self.viewer.sync()
                time.sleep(TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)
                
    def step(self, animate: bool = True):
        mujoco.mj_step(self.model, self.data)
        if animate and self.viewer:
            self.viewer.sync()
            time.sleep(TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)
    
    # def shake_gripper(self, animate: bool = True):
    #     # axis letter -> unit vector
    #     AX = {
    #         'x': np.array([1.0, 0.0, 0.0]),
    #         'y': np.array([0.0, 1.0, 0.0]),
    #         'z': np.array([0.0, 0.0, 1.0]),
    #     }

    #     # quaternion multiply (wxyz)
    #     def qmul(a, b):
    #         aw, ax, ay, az = a
    #         bw, bx, by, bz = b
    #         return np.array([
    #             aw*bw - ax*bx - ay*by - az*bz,
    #             aw*bx + ax*bw + ay*bz - az*by,
    #             aw*by - ax*bz + ay*bw + az*bx,
    #             aw*bz + ax*by - ay*bx + az*bw
    #         ], dtype=float)

    #     # axis-angle -> quat (wxyz)
    #     def axis_angle_quat(axis, angle):
    #         half = 0.5 * angle
    #         s = np.sin(half)
    #         return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)

    #     p0 = self.default_gripper_pose.copy()
    #     q0 = self.default_gripper_orient.copy()

    #     n_shake      = env_consts.SHAKING_PARAMETERS['n_shake']
    #     t_per_target = env_consts.SHAKING_PARAMETERS.get('t_cmd_stable', 50)
    #     ids          = env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']

    #     for gid in ids:
    #         info   = sh_consts.GRIPPER_6DOF_INFOS[gid]
    #         axis   = AX[info['axis']]
    #         typ    = info['type']              # 'prismatic' or 'revolute'
    #         amp    = float(info['joint_target_val'])

    #         # sequence like Bullet: +amp, -amp, 0 around the saved default
    #         for _ in range(n_shake):
    #             for val in ( amp, -amp, 0.0 ):
    #                 for _ in range(t_per_target):
    #                     if typ == 'prismatic':
    #                         pos  = p0 + axis * val
    #                         quat = q0
    #                     else:  # revolute
    #                         dq   = axis_angle_quat(axis, val)  # unit quaternion
    #                         quat = qmul(q0, dq)
    #                         pos  = p0

    #                     self.set_6dof_pose_gripper(pos, quat)
    #                     self.step(animate)

    #     # return to exact default
    #     self.reset_gripper_pose()
    #     self.step(animate)


    def shake_gripper(self, animate: bool = True):
        p0 = self.default_gripper_pose.copy()
        q0 = self.default_gripper_orient.copy()

        amp = 0.2
        n_shake = env_consts.SHAKING_PARAMETERS['n_shake']
        hold = env_consts.SHAKING_PARAMETERS.get('t_cmd_stable', 50)

        # sequence like Bullet: +A, -A, 0 around default
        targets = [ +amp, -amp, 0.0 ]

        for _ in range(n_shake):
            for dx in targets:
                p_goal = p0 + np.array([dx, 0.0, 0.0])    # translate along X only
                # linearly ramp from current pose to goal over `hold` frames
                # start from whatever we last set (use current qpos as start)
                p_start = self.data.mocap_pos[self.model.body_mocapid[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
                ]].copy()

                for k in range(hold):
                    t = (k + 1) / float(hold)
                    p = (1.0 - t) * p_start + t * p_goal
                    self.set_6dof_pose_gripper(p, q0)   # keep orientation fixed
                    self.step(animate)

        # snap back to exact default and hold a bit
        self.set_6dof_pose_gripper(p0, q0)
        for _ in range(hold):
            self.step(animate)


    # def shake_gripper(self, animate: bool = True):      
    #     n_shake = env_consts.SHAKING_PARAMETERS['n_shake']
    #     i_shake = 0
    #     target_position = 0.5
        
    #     target_j_poses = [target_position, -target_position, 0]
    #     while i_shake < n_shake:
    #         for j_pose in target_j_poses:
    #             for _ in range(env_consts.SHAKING_PARAMETERS['t_cmd_stable']):
    #                 pose = [0, 0, j_pose]
    #                 self.set_6dof_pose_gripper(pose, self.default_gripper_orient)
    #                 self.step(animate)
    #         i_shake += 1
        
        
        # for i_grip_joint in env_consts.SHAKING_PARAMETERS['perturbated_joint_ids']:
        #     gripper_joint_infos = sh_consts.GRIPPER_6DOF_INFOS[i_grip_joint]
            
            
        #     target_position = 1 # gripper_joint_infos['joint_target_val']
        #     velocity = gripper_joint_infos['max_vel']
        #     type = gripper_joint_infos['type']
        #     axis = gripper_joint_infos['axis']
            
        #     i_shake = 0

            # target_j_poses = [target_position, -target_position, 0]
            # while i_shake < n_shake:
            #     for j_pose in target_j_poses:
            #         for _ in range(env_consts.SHAKING_PARAMETERS['t_cmd_stable']):
            #             pose = [j_pose, 0, 0]
            #             self.set_6dof_pose_gripper(pose, self.default_gripper_orient)
            #             self.step(animate)
            #     i_shake += 1

        # self.reset_gripper_pose()
        # self.step(animate)
  

        
        