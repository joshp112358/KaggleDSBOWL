from common import *


## for debug
def dummy_transform(image):
    print ('\tdummy_transform')
    return image


# geometric augment
def fix_resize_transform2(image, mask, w, h):
    image = cv2.resize(image,(w,h))
    mask  = cv2.resize(mask,(w,h))
    mask  = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    return image, mask




# geometric augment
def fix_resize_transform(image, w, h):
    image = cv2.resize(image,(w,h))
    return image



def random_horizontal_flip_transform(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img) ##left-right
    return image



def random_rotate90_transform(image, u=0.5):
    if random.random() < u:

        angle=random.randint(1,3)*90
        if angle == 90:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,1)
            #return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            image = cv2.flip(image,-1)
            #return img[::-1,::-1,:]
        elif angle == 270:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,0)
            #return  img.transpose((1,0, 2))[::-1,:,:]

    return image

#
# def random_shift_scale_rotate(image, shift_limit=[-0.0625,0.0625], scale_limit=[1/1.2,1.2],
#                                rotate_limit=[-15,15], aspect_limit = [1,1],  size=[-1,-1], borderMode=cv2.BORDER_REFLECT_101 , u=0.5):
#     #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT
#
#     if random.random() < u:
#         height,width,channel = image.shape
#         if size[0]==-1: size[0]=width
#         if size[1]==-1: size[1]=height
#
#         angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
#         scale  = random.uniform(scale_limit[0],scale_limit[1])
#         aspect = random.uniform(aspect_limit[0],aspect_limit[1])
#         sx    = scale*aspect/(aspect**0.5)
#         sy    = scale       /(aspect**0.5)
#         dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
#         dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)
#
#         cc = math.cos(angle/180*math.pi)*(sx)
#         ss = math.sin(angle/180*math.pi)*(sy)
#         rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])
#
#         box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
#         box1 = box0 - np.array([width/2,height/2])
#         box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
#
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0,box1)
#
#         image = cv2.warpPerspective(image, mat, (size[0],size[1]),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
#
#     return image
#
#
#
#
#
# def random_crop_scale (image, scale_limit=[1/1.2,1.2], size=[-1,-1], u=0.5):
#     if random.random() < u:
#         image=image.copy()
#
#         height,width,channel = image.shape
#         sw,sh = size
#         if sw==-1: sw = width
#         if sh==-1: sh = height
#         box0 = np.array([ [0,0], [sw,0],  [sw,sh], [0,sh], ])
#
#         scale = random.uniform(scale_limit[0],scale_limit[1])
#         w = int(scale * sw)
#         h = int(scale * sh)
#
#         if w>width and h>height:
#             x0 = random.randint(width-w, 0)
#             y0 = random.randint(height-h,0)
#             x1 = x0+w
#             y1 = y0+h
#         elif w<width and h<height:
#             x0 = random.randint(0, width-w)
#             y0 = random.randint(0, height-h)
#             x1 = x0+w
#             y1 = y0+h
#         else:
#             raise NotImplementedError
#             #pass
#
#         box1 = np.array([ [x0,y0], [x1,y0],  [x1,y1], [x0,y1], ])
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box1,box0)
#
#         #<debug>
#         # cv2.rectangle(image,(0,0),(width-1,height-1),(0,0,255),1)
#         # cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,0),3)
#         # print(x0,y0,x1,y1)
#         image = cv2.warpPerspective(image, mat, (sw,sh),flags=cv2.INTER_LINEAR, #cv2.BORDER_REFLECT_101
#                                     borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
#
#     return image





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    print('\nsucess!')