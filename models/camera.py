import numpy as np
import torch 


class Camera():
    def __init__(self, hfov, res):
        """
        Defines a PPC (camera) that we will be using to model the world and light. 
        """
        self.camera = self.initialize_camera_matrix(hfov, res)
        self.res = res

        """
        Blender to standard coordinate system transform. (ours)
        Standard coordinate system is: x right y up z out (out=screen to face)
        Blender coordinate system is: x right y in z up
        """
        self._coord_trans = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)

    def initialize_camera_matrix(self, hfov, res):
        """
        Returns a 3x3 Matrix 
        M = [a,b,c]; {a,b,c} \in 3x1 matrix
        """
        w,h = res 
        hfov = torch.tensor(hfov)
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, -1.0, 0.0])
        hfovd = hfov / torch.tensor(180.0) * np.pi
        c = torch.tensor([-w / 2.0, h / 2, -w / (2 * torch.tan(hfovd / 2.0))])
        return torch.stack([a,b,c]).T # (3,3)

    def get_a(self,):
        return self.camera[:,0]

    def get_b(self,):
        return self.camera[:,1]

    def get_c(self,):
        return self.camera[:,2]

    @classmethod
    def from_camera_eyepos(cls, eye_pos, camera):
        c_ = cls(30, (400,400))
        c_.res = None
        c_.camera = camera 
        c_.eye_pos = eye_pos
        return c_ 
        
    @staticmethod
    def c2w_from_lookat(eye_pos, look_at_point, up_guidance=np.array([0, 1, 0], dtype=np.float32)):
        """
        Get 4x4 camera to world space matrix, for camera at eye_pos looking at look_at_point
        """
        back = eye_pos - look_at_point
        back /= np.linalg.norm(back)
        right = np.cross(up_guidance, back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)

        cam_to_world = np.empty((4, 4), dtype=np.float32)
        cam_to_world[:3, 0] = right
        cam_to_world[:3, 1] = up
        cam_to_world[:3, 2] = back
        cam_to_world[:3, 3] = eye_pos
        cam_to_world[3, :] = [0, 0, 0, 1]
        return cam_to_world

    def set_pose_using_blender_matrix(self, c2w, transform_coords=False):
        """
        c2w: (type tensor) [3,4] doesn't know what the resolution 
        of the image is (has the extrinsic parameters)
        pixels = K @ c2w @ p -> ppc @ p 
        """
        if transform_coords:
            # print("self.eye_pos", c2w[:, 3])
            # eye = torch.cat([c2w[:, 3].float(), torch.tensor([1])])
            # self.eye_pos = eye @ self._coord_trans
            # self.eye_pos = self.eye_pos[:3]
            # # print("self.eye_pos", self.eye_pos)
            # c2w_t = c2w.float() @ self._coord_trans
            # # print(c2w_t)
            # self.camera = c2w_t[:, :3].float() @ self.camera.float()
            # print(self.camera)
            raise ValueError("This is not needed anymore. please do not use this flag.")
        else:
            self.eye_pos = c2w[:, 3].float() # Camera location c2w (x,y,z) vs. (x,z,-y)
            self.camera = c2w[:, :3].float() @ self.camera.float() 
            # print(self.camera, self.eye_pos)

    def set_camera_matrix(self, eye_pos, lookAtPoint, upGuidance):
        """
        Sets the camera matrix, eye_pos 
        M = [a,b,c]; {a,b,c} \in 3x1 matrix 
        Args: res, eye_pos, lookAtPoint, upGuidance 
        """
        w,h = self.res
        self.upGuidance = torch.tensor(upGuidance)
        self.lookAtPoint = torch.tensor(lookAtPoint)
        self.eye_pos = torch.tensor(eye_pos)

        diff = self.lookAtPoint - self.eye_pos
        norm = torch.linalg.norm(diff)
        newvd = diff/norm

        cross = torch.cross(newvd, self.upGuidance, axis=-1)
        cross_norm = torch.linalg.norm(cross)
        newa = cross/cross_norm

        cross = torch.cross(newvd, newa, axis=-1)
        cross_norm = torch.linalg.norm(cross)    
        newb = cross/cross_norm

        cross = torch.cross(self.camera[:,0], self.camera[:,1])
        cross_norm = torch.linalg.norm(cross)
        focalLength = torch.dot(cross/cross_norm, self.camera[:,2])
        newc = newvd*focalLength - newa*w / 2.0 - newb*h / 2.0

        self.camera = torch.stack([newa,newb,newc]).T #  matrix (3,3)

    def get_transformation_to(self, to_camera, device='cpu'): 
        """
        returns R,Q matrix for transformation from self to camera
        to_camera: expects the camera class; eg. light presp.
        """
        O_minus_L = self.eye_pos.to(device) - to_camera.eye_pos.to(device)
        O_minus_L = O_minus_L.to(device)
        ML_inv = torch.inverse(to_camera.camera).to(device)
        # print("get_transformation_to ML, O_minus_L", ML_inv.shape, O_minus_L.shape)
        Q = ML_inv @ O_minus_L
        R = ML_inv @ self.camera.to(device)
        return R, Q #[R, Q]
