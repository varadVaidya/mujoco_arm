from mujoco_arm.robot import MujocoArm
from time import sleep

if __name__ == "__main__":
    
    UR5_PATH = "assets/universal_robots_ur5e/scene.xml"
    PANDA_PATH = "assets/franka_emika_panda/scene.xml"
    
    arm = MujocoArm(PANDA_PATH)
    arm._viewer.viewer.render()
    
    sleep(5)
    
    
