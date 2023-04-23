#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import cv2
import matplotlib.pyplot as plt

def prediction(model, image, patch_size = (2200,1550)):
    segm_img = torch.zeros(image.shape)  ##this is 2d, only one channel.  do i want (x,y,1)?
    patch_w, patch_h = patch_size
    #print(segm_img.shape)
    patch_num=1
    img_w_shape = image.shape[2]
    img_h_shape = image.shape[3]
    for i in range(0, img_w_shape, patch_w):   #step by the width of the patch
        for j in range(0, img_h_shape, patch_h):  #step by height of patch
            w_start = i
            h_start = j
            w_end = i+patch_w
            h_end = j+patch_h
            #print(img_w_shape,w_end,img_h_shape,h_end )
            if w_end > img_w_shape:
              w_end = img_w_shape
              w_start = w_end - patch_w
            if h_end > img_h_shape:
              h_end = img_h_shape
              h_start = h_end - patch_h


            single_patch = image[:,:,w_start:w_end, h_start:h_end]
            single_patch_shape = single_patch.shape
            
            
            single_patch_prediction = single_patch ##single_patch is input to model
            
            segm_img[:,:,w_start:w_start+patch_w, h_start:h_start+patch_h] += single_patch_prediction
          
            #print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img




def display_depth_image(depth_img, scale = 1, show_image = True, target_mask = False):
  """
  makes the depth predictions into an image.
  blue is farthest away, red is closest.  
  
  depth_img is either a target or a model output.  should be 1 channel
  scale is a constant and is multiplied by the inputs.  just used to map to rgb
  show_image will print the image for convenience.  
  target_mask if it is set to true, it makes all the pixels that are exaclty zero white:
  per the competition those pixels are errors, not actual depth readings.  
  """
  b = depth_img
  rel_max = torch.max(b)
  g = torch.zeros_like(depth_img)
  r = rel_max - b
  pic = torch.cat((r,g,b),axis = 1)

  if target_mask:
    #for targets all 0s are invalid, make em white
    target_mask = depth_img == 0
    target_mask = target_mask.expand(-1,3,-1,-1)
    pic[target_mask] = 255

  pic = pic.squeeze().permute(1,2,0) * scale
  
  if show_image:
    plt.imshow(pic)
  return pic

