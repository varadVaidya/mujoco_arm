import numpy as np
import mujoco
from mujoco import viewer
import mujoco_viewer
import PIL.Image
import cv2

###! TO BE DECIDED HOW TO BE USED HERE


class Renderer():
    
    def __init__(self,
                 model, data,
                 mode='mujoco_viewer',
                 width=1280,
                 height=720,
                 save_video=False,
                 video_name=None,
                 FPS = 60.0
                 ):
        
        self._mode_list = ['mujoco','mujoco_viewer','offscreen']
        
        if mode not in self._mode_list:
            # raise value error
            raise NotImplementedError("selected mode not among ['mujoco','mujoco_viewer','offscreen']")
        
        self._model = model
        self._data = data
        
        self.mode = mode
        
        self.width = width
        self.height = height
        self.video_name = video_name
        self.FPS = FPS
        if self.mode == 'offscreen':
            self.viewer = mujoco_viewer.MujocoViewer(self._model, self._data,
                                                     'offscreen',
                                                     width = self.width, height = self.height)
            self.rendered_frames = []
        
        if self.mode == 'mujoco_viewer':
            self.viewer = mujoco_viewer.MujocoViewer(self._model, self._data,
                                                     width=self.width,height=self.height)
        
        if self.mode == 'mujoco':
            self.viewer = viewer.launch(self._model, self._data)
    
    def update(self):
        
        if self.mode =='offscreen':
            frame = self.viewer.read_pixels()
            frame = PIL.Image.fromarray(frame)
            
            self.rendered_frames.append(frame.copy())
        
        if self.mode == 'mujoco_viewer':
            self.viewer.render()
        
        if self.mode == 'mujoco':
            pass

    def write_frames_to_video(self,frames=None,FPS=60.0):
        
        if frames is None:
            frames = self.rendered_frames
        if self.video_name is None:
            self.video_name = "mujoco_sim.mp4"
        width, height = frames[0].size
        video_dims = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.video_name, fourcc, self.FPS, video_dims)
        
        for frame in frames:
            video.write(cv2.cvtColor(np.array(frame) , cv2.COLOR_BGR2RGB))
        
        video.release()


        
        
        
        