import mujoco
import numpy as np
from typing import Tuple

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
        
    def reset(self):
        self.reset_gripper_pose()
        self.reset_object_pose()
    
    def step(self, n: int = 1, sync: bool = False):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if sync and self.viewer is not None:
                self.viewer.sync()
                 
    def get_actuator_info(self, actuator_name: str) -> dict:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        lo, hi   = self.model.actuator_ctrlrange[aid]
        flo, fhi = self.model.actuator_forcerange[aid]

        return {
            "aid": aid,
            "low": float(lo),
            "high": float(hi),
            "force_low": float(flo),
            "force_high": float(fhi),
        }
        
    def drive_actuator_to_max(self, actuator_name: str, steps: int, animate: bool) -> None:
        info = self.get_actuator_info(actuator_name)
        aid = info["aid"]
        target = info["high"]
        self.data.ctrl[aid] = target
        self.step(steps, sync=animate)