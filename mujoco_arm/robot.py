import copy
import time
import logging
import threading
import quaternion
import numpy as np
import mujoco as mj
from mujoco import viewer
from mujoco_arm.utils.render_class import Renderer
from threading import Lock


LOG_LEVEL = "DEBUG"

class MujocoArm():
    
    """
    put docstring later
    
    """
    
    def __init__(self, model_path,
                 render = True, config = None):
        
        logging.basicConfig(format='\n{}: %(levelname)s: %(message)s\n'.format(
            self.__class__.__name__), level=LOG_LEVEL)
        
        self._logger = logging.getLogger(__name__)
        
        self._logger.debug("Initializing MujocoArm \n")
        
        self._model = mj.MjModel.from_xml_path(model_path)
        self._logger.debug("Loaded model from " + str(model_path))
        self._data = mj.MjData(self._model)
        
        mj.mj_forward(self._model, self._data)
        
        if render:
            self._viewer = Renderer(self._model, self._data)
        else:
            self._viewer = None
        
        self._has_gripper = False 
        
        self._define_joint_ids()
    
    def _define_joint_ids(self):
        # transmission type (0 == joint)
        trntype = self._model.actuator_trntype
        # transmission id (get joint actuated)
        trnid = self._model.actuator_trnid

        ctrl_joints = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i, 0] not in ctrl_joints:
                ctrl_joints.append(trnid[i, 0])

        self.controllable_joints = sorted(ctrl_joints)
        self._logger.debug("Controllable joints: " + str(self.controllable_joints))
        self.movable_joints = self._model.jnt_dofadr
        self.qpos_joints = self._model.jnt_qposadr
        
        
        
        
        
    
    
    