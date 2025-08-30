import mujoco
import numpy as np
from typing import Tuple
import time

import environments.src.robots.mj_shadow_hand_consts as sh_consts

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


    def close_viewer(self):
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
    def set_6dof_pose_gripper(self, pos_xyz, orient_quat):
        jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        adr  = self.model.jnt_qposadr[jid]   # start index in qpos for this freejoint (7 slots: xyz + quat)
        vadr = self.model.jnt_dofadr[jid]    # start index in qvel (6 slots)
        
        self.data.qpos[adr:adr+3] = pos_xyz
        self.data.qpos[adr+3:adr+7] = orient_quat
        self.data.qvel[vadr:vadr+6] = 0  # Zero base velocities so we don't inject impulses
        mujoco.mj_forward(self.model, self.data)  # Recompute derived quantities (contacts, kinematics, etc.)
        
    def set_6dof_pose_object(self, pos_xyz, orient_quat):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        qadr = self.model.jnt_qposadr[jid]
        vadr = self.model.jnt_dofadr[jid]
        
        self.data.qpos[qadr:qadr+3] = pos_xyz
        self.data.qpos[qadr+3:qadr+7] = orient_quat
        self.data.qvel[vadr:vadr+6] = 0
        mujoco.mj_forward(self.model, self.data)
        
    def reset_gripper_pose(self):
        self.set_6dof_pose_gripper(self.default_gripper_pose, self.default_gripper_orient)
        
    def reset_object_pose(self):
        self.set_6dof_pose_object(self.default_object_pose, self.default_object_orient)
    
    # def reset_robot_fingers(self):
    #     for actuator_name, default_val in sh_consts.DEFAULT_JOINT_STATES.items():
    #         aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    #         self.data.ctrl[aid] = default_val

    #     mujoco.mj_forward(self.model, self.data)
        
    #     if self.viewer:
    #         self.viewer.sync()
            
    def reset_robot_fingers(self):
        m, d = self.model, self.data

        for actuator_name, default_val in sh_consts.DEFAULT_JOINT_STATES.items():
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            if actuator_name in sh_consts.TENDON_TO_JOINTS:
                    joint_names = sh_consts.TENDON_TO_JOINTS[actuator_name]
                    share = float(default_val) / len(joint_names) # split the desired tendon target evenly across its joints
                    for jname in joint_names:
                        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        qadr = m.jnt_qposadr[jid]
                        dadr = m.jnt_dofadr[jid]
                        d.qpos[qadr] = share
                        d.qvel[dadr] = 0.0
            else:
                target_id = int(m.actuator_trnid[aid, 0])
                qadr = m.jnt_qposadr[target_id]
                dadr = m.jnt_dofadr[target_id]
                d.qpos[qadr] = float(default_val)
                d.qvel[dadr] = 0.0

        mujoco.mj_forward(m, d)
        
        if self.viewer:
            self.viewer.sync()

             
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
            
            

    # def get_actuator_info(self, actuator_name: str) -> dict:
    #     aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    #     lo, hi   = self.model.actuator_ctrlrange[aid]
    #     flo, fhi = self.model.actuator_forcerange[aid]

    #     return {
    #         "aid": aid,
    #         "low": float(lo),
    #         "high": float(hi),
    #         "force_low": float(flo),
    #         "force_high": float(fhi),
    #     }
            
    # def close_gripper(self, actuator_name: str):                
    #     info = self.get_actuator_info(actuator_name)
    #     aid = info["aid"]
    #     target = info["high"]
        
    #     for _ in range(MAX_STEP_CLOSE_GRIP):
    #         self.data.ctrl[aid] = target
    #         mujoco.mj_step(self.model, self.data)
            
    #         if self.viewer:
    #             self.viewer.sync()
    #             time.sleep(TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)