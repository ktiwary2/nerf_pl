        shadow_maps = []
        curr_eye_pos = ppc['eye_pos'][0]
        prev_split_at = 0
        curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][0].squeeze(0), camera=ppc['camera'][0].squeeze(0))
        for i in range(len(ppc['camera'])):
            # each pixel can have a different viewpoint within the batch, therefore we need to split them with the same (depth, pose) 
            # print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
            if not (torch.equal(curr_eye_pos, ppc['eye_pos'][i])): 
                # means a new ppc is encountered 
                # all pixels from prev_split to i have the same camera pose therefore we can estimate the sm 
                # for those.
                print("------")
                print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
                print("Found different eye_pos, using shadow method {}, splitting at {}:{}".format(shadow_method, prev_split_at, i))
                sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:i,:]
                sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, sub_batch_mesh_range_cam, 
                                           meshed_normed_light_cam, sub_batch_mesh_range_cam.device, mode=shadow_method, \
                                           delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                           use_numpy_meshgrid=True)
                print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
                print("------")

                shadow_maps += [sm]
                prev_split_at = i 
                curr_eye_pos = ppc['eye_pos'][i]
                curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][i].squeeze(0), camera=ppc['camera'][i].squeeze(0))

        if prev_split_at == 0: 
            print("Found No Splits...")
            # means that all the pixels have the same viewpoint! we can batch them together
            shadow_maps = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, batched_mesh_range_cam, 
                                        meshed_normed_light_cam, batched_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True) # (num_rays, 3)
        else: #prev_split_at < (len(ppc)-1): 
            # do inference on the remaining 
            print("Doing inference on the last split from [{}:{}]".format(prev_split_at, len(ppc['camera'])))
            sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:,:]
            sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, sub_batch_mesh_range_cam, 
                                        meshed_normed_light_cam, sub_batch_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True) # (num_rays, 3)
            print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
            shadow_maps += [sm]
            shadow_maps = torch.cat(shadow_maps, 0) # (num_rays, 3)
            print("shadow_maps.shape", shadow_maps.shape)



   def inference(ppc, light_camera, image_shape, batched_mesh_range_cam, meshed_normed_light_cam, shadow_method):
        """
        ppc: [Batch_size] Camera Poses: instance of the Camera() class
        light_camera: Instance of class Camera placed at the light position 
        batch_size: batch_size
        batched_mesh_range_cam: [num_rays, 4] (i, j, 1, depth] from camera viewpoints
        meshed_normed_light_cam: [H*W, 4] (i, j, 1, depth] from light viewpoint
        """
        shadow_maps = []
        import time
        t = time.time()
        for i in range(len(ppc['camera'])):
            curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][i].squeeze(0), camera=ppc['camera'][i].squeeze(0))
            # Do pixel by Pixel projection and test whether they are in shadow (see extras.py for faster impl.)
            # currently doesn't help since te rays are randomized! 
            # print("batched_mesh_range_cam[i,:].shape", batched_mesh_range_cam[i,:].shape)
            sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, batched_mesh_range_cam[i,:].view(-1,4), 
                                        meshed_normed_light_cam, batched_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True)
            # print("Sm: {}".format(sm.shape))
            shadow_maps += [sm]

        shadow_maps = torch.cat(shadow_maps, 0) # (num_rays, 3)
        # print("shadow_maps.shape", shadow_maps.shape)
        print("time:", time.time()-t)
        return shadow_maps # (num_rays)