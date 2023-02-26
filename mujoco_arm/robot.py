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
    'ft_site_name': 'attachment_site',
    # 'ee_name': 'attachment_site'
}


LOG_LEVEL = "DEBUG"


class MujocoArm():

    """
    put docstring later

    """

    def __init__(self, model_path,
                 render=True, config=DEFAULT_CONFIG,
                 prestep={}, poststep={}):

        logging.basicConfig(format='\n{}: %(levelname)s: %(message)s\n'.format(
            self.__class__.__name__), level=LOG_LEVEL)

        self._logger = logging.getLogger(__name__)

        self._logger.debug("Initializing MujocoArm \n")

        self._model = mj.MjModel.from_xml_path(model_path)
        self._logger.debug("Loaded model from " + str(model_path))
        self._data = mj.MjData(self._model)

        mj.mj_forward(self._model, self._data)

        if render:
            self._viewer = Renderer(
                self._model, self._data, mode='offscreen')
        else:
            self._viewer = None

        self._has_gripper = False

        self._define_joint_ids()

        self._nq = len(self.qpos_joints)  # ^ total number of joints
        self._nu = len(self.controllable_joints)  # ^ number of actuators.

        self._all_joint_names = [
            self._model.joint(j).name for j in self.qpos_joints
        ]

        self._all_joint_dict = dict(
            zip(self._all_joint_names, self.qpos_joints)
        )

        self._site_names = [
            self._model.site(i).name for i in range(self._model.nsite)
        ]

        self._body_names = [
            self._model.body(i).name for i in range(self._model.nbody)
        ]

        if 'ee_name' in config:
            self.set_as_ee(config['ee_name'])  # ! TODO: define this function
        else:
            # ! TODO: define this function
            self._ee_id, self._ee_name = self._use_last_defined_link()
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

    def set_as_ee(self, body_name):
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
            "Using last defined link as end effector: " +
            str(name) + " with id: " + str(id)
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
        self._logger.debug("Controllable joints: " +
                           str(self.controllable_joints))
        # & jnt_dofadr == some type of adress
        self.movable_joints = self._model.jnt_dofadr
        self._logger.debug("Movable Joints are: " + str(self.movable_joints))
        self.qpos_joints = self._model.jnt_qposadr
        self._logger.debug("qpos Joints are: " + str(self.qpos_joints))

        # ^ qpos joints is used in calculating total number of joints, joint name and
        # ^ joint position.
        # ^ jnt_dofadr is used in calculating joint velocities, and joint acceleration

        # & according to the documentation, it seems that qpos and qvel are stored in
        # & the in the same array for all the bodies present in the simulation (???) So we
        # & use jnt_dofadr and jnt_qposadr for adress for joint position and velocity.

    @ property
    def sim(self):
        """
        Get the mujoco simulation object
        """
        raise NotImplementedError()

    @ property
    def viewer(self):
        """
        Get the mujoco viewer object
        """
        return self._viewer

    @ property
    def model(self):
        """
        Get the mujoco model object
        """
        return self._model

    def has_body(self, bodies):
        """
        Check if the provided bodies exist in the model.

        :param bodies: list of body names
        :type bodies: [str]
        :return: True if all bodies exist, False otherwise
        :rtype: bool
        """

        if isinstance(bodie, str):
            bodies = [bodies]

        for body in bodies:
            if body not in self._body_names:
                return False
        return True

    def get_ft_reading(self, in_global_frame=True):
        """Retuen sensor data values. Assumems no other senor is present. #^ questionable assumption

        Args:
            in_global_frame (bool, optional): if the retuened force values have to be in global frame of reference.
            Defaults to True.

        Raises:
            NotImplementedError: Not implemented because not found use till now.
        """
        raise NotImplementedError(
            "NOT IMPLEMENTED BECAUSE NOT FOUND USE TILL NOW")

    def get_contact_info(self):
        """
        Get detauils about physical contacts between bodies.
        Includes contact point position, orientation, contact force.

        :return: list of ContactInfo objects. #^ Not sure what this is right now...
        :rtype: [ContactInfo]
        """
        raise NotImplementedError()

    def body_jacobian(self, body_id=None, joint_indices=None, recompute=True):
        """ retuen body jacobian at the current step.

        Args:
            body_id (int, optional): id of body whose jacobian is to be computed.
            Defaults to end effector.

            joint_indices ([int], optional): list of joint indices, deafults to all controllable joints.

            recompute (bool, optional): if set to true, will perform forward kinematics computation
            for the step and provide updated values. Defaults to True.

        Returns:
            ndarray: 6xN array, as body jacobian.
        """
        if body_id is None:
            body_id = self._ee_id

            if self._ee_is_site:
                body_id = self._model.site_id2bodyid(body_id)
                return self.site_jacobian(body_id, joint_indices)

        if joint_indices is None:
            joint_indices = self.controllable_joints

        if recompute and not self._forwarded:
            self.forward_sim()  # ^ this is the forward kinematics computation.
            #! NOT IMPLEMENTED YET

        position_jac = np.zeros((3, self._model.nv))
        orientation_jac = np.zeros((3, self._model.nv))

        body_pos, body_ori = self.body_pose(body_id)

        mj.mj_jac(self._model, self._data, position_jac,
                  orientation_jac, body_pos, body_id)

        body_jacobian = np.vstack((position_jac, orientation_jac))

        assert body_jacobian.shape == (6, self._model.nv)

        return body_jacobian

    def site_jacobian(self, site_id, joint_indices=None, recompute=True):
        """return jacobian computed for a site defined in the model.

        Args:
            site_id (int): index/id of the site.
            joint_indices ([int], optional): list of joint indices. Defaults to all movable joints.
            recompute (bool, optional): if set to True, will perform forward kinematics computation
            for the step, and provide updated results. Defaults to True.

        Returns:
            ndarray: 6xN array, as site jacobian.
        """

        if joint_indices is None:
            joint_indices = self.controllable_joints

        if recompute and not self._forwarded:
            self.forward_sim()

        position_jac = np.zeros((3, self._model.nv))
        orientation_jac = np.zeros((3, self._model.nv))

        site_pos, site_ori = self.site_pose(site_id)

        mj.mj_jac(self._model, self._data, position_jac,
                  orientation_jac, site_pos, site_ori)

        site_jacobian = np.vstack((position_jac, orientation_jac))

        assert site_jacobian.shape == (6, self._model.nv)

        return site_jacobian

    def ee_pose(self):
        """return end effector pose.

        Returns:
            ndarray,ndarray: EE position (x,y,z). EE quaternion (w,x,y,z)
        """
        if self._ee_is_site:
            return self.site_pose(self._ee_id)

        return self.body_pose(self._ee_id)

    def body_pose(self, body_id, recompute=True):
        """ return pose of the body at curent time step.

        Args:
            body_id (int): id of the body.
            recompute (bool, optional):if set to True, will perform forward kinematics computation
            for the step, and provide updated results. Defaults to True.

        Returns:
            ndarray,ndarray: body position (x,y,z). body quaternion (w,x,y,z)
        """

        if recompute and not self._forwarded:
            self.forward_sim()

        body_pos = self._data.xpos[body_id].copy()
        body_quat = self._data.xquat[body_id].copy()

        return body_pos, body_quat

    def site_pose(self, site_id, recompute=True):
        """ return pose of the site at curent time step.

        Args:
            site_id (int): id of the site.
            recompute (bool, optional):if set to True, will perform forward kinematics computation
            for the step, and provide updated results. Defaults to True.

        Returns:
            ndarray,ndarray: site position (x,y,z). site quaternion (w,x,y,z)
        """
        if recompute and not self._forwarded:
            self.forward_sim()
        site_pos = self._data.site_xpos[site_id].copy()
        site_quat = self._data.site_xquat[site_id].copy()

        return site_pos, site_quat

    def forward_sim(self):
        """Perform forward kinematics computation for the current step."""
        mj.mj_forward(self._model, self._data)
        self._forwarded = True
