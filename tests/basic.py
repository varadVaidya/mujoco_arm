from mujoco_arm.robot import MujocoArm
from time import sleep

if __name__ == "__main__":

    UR5_PATH = "assets/universal_robots_ur5e/scene.xml"
    PANDA_PATH = "assets/franka_emika_panda/scene.xml"

    arm = MujocoArm(PANDA_PATH)
    arm._viewer.update()
    arm.forward_sim()
    pos, ori = arm.ee_pose()
    ee_jac = arm.body_jacobian()

    print("Position: ", pos)
    print("Orientation: ", ori)

    print("Jacobian: \n", ee_jac)
    sleep(2)
