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

DEFAULT_CONFIG = {
    'ft_site_name': 'ee_site',
    # 'ee_name': 'attachment_site'
}


LOG_LEVEL = "DEBUG"

class MujocoArm():
    
    """
    put docstring later
    
    """
    
    def __init__(self, model_path,
                 render = True, config = DEFAULT_CONFIG,
                 prestep = {}, poststep = {}):
        
        logging.basicConfig(format='\n{}: %(levelname)s: %(message)s\n'.format(
            self.__class__.__name__), level=LOG_LEVEL)
        
        self._logger = logging.getLogger(__name__)
        
        self._logger.debug("Initializing MujocoArm \n")
        
        self._model = mj.MjModel.from_xml_path(model_path)
        self._logger.debug("Loaded model from " + str(model_path))
        self._data = mj.MjData(self._model)
        
        mj.mj_forward(self._model, self._data)
        
        if render:
            self._viewer = Renderer(self._model, self._data,mode='mujoco_viewer')
        else:
            self._viewer = None
        
        self._has_gripper = False 
        
        self._define_joint_ids()
        
        self._nq = len(self.qpos_joints)    #^ total number of joints
        self._nu = len(self.controllable_joints) #^ number of actuators.
        
        self._all_joint_names = [
            self._model.joint(j).name for j in self.qpos_joints
        ]
        
        self._all_joint_dict = dict(
            zip(self._all_joint_names, self.qpos_joints)
        )
        
        self._site_names = [
            self._model.site(i).name for i in range(self._model.nsite)
        ]
        
        if 'ee_name' in config:
            self.set_as_ee(config['ee_name']) #! TODO: define this function
        else:
            self.__ee_id,self._ee_name = self._use_last_defined_link() #! TODO: define this function
            self._ee_is_site = False
        
        if 'ft_site_name' in config:
            self._ft_site_name = config['ft_site_name']
        else:
            self._ft_site_name = False
        
        self._mutex = Lock()
        self._asynch_thread_active = False
        self._forwarded = False
        
        self._pre_step_callables = prestep
        self._post_step_callables = poststep
        
        self._first_step_not_done = True
        
    def set_as_ee(self,body_name):
        """
        Set the provided body or site as the end_effector of the robot.
        
        sets the id of the end effector, for further calculations.
        :param body_name: name of the body or site in mujoco model.
        :tyep body_name: str
        """
        
        self._ee_name = body_name
        
        if body_name in self._site_names:
            self._ee_is_site = True
            self._ee_id = self._model.site(body_name).id
            self._logger.debug(
                "End effecter is a site in model: " + str(body_name)
            )
        else:
            self._ee_is_site = False
            self._ee_id = self._model.body(body_name).id
            self._logger.debug(
                "End effecter is a body in model: " + str(body_name)
            )
        
    def _use_last_defined_link(self):
        """
        use the last defined link in the model, and sets it as the end effector.
        
        sets the id and the name of the end effector.        
        """
        id = self._model.nbody - 1
        name = self._model.body(id).name
        self._logger.debug(
            "Using last defined link as end effector: " + str(name) + " with id: " + str(id)
        )
        return id, name
    
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
        self.movable_joints = self._model.jnt_dofadr        #& jnt_dofadr == some type of adress
        self._logger.debug("Movable Joints are: " + str(self.movable_joints))
        self.qpos_joints = self._model.jnt_qposadr
        self._logger.debug("qpos Joints are: " + str(self.qpos_joints))
        
        #^ qpos joints is used in calculating total number of joints, joint name and
        #^ joint position.
        #^ jnt_dofadr is used in calculating joint velocities, and joint acceleration
        
        #& according to the documentation, it seems that qpos and qvel are stored in
        #& the in the same array for all the bodies present in the simulation (???) So we
        #& use jnt_dofadr and jnt_qposadr for adress for joint position and velocity.
        
        
    
        
        
        
        
        
        
    
    
    