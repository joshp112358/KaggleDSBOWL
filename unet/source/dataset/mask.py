from common import *

from utility.file import *
from utility.draw import *


## mask ## "run-length encoding" !! 1-index

#https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode0(mask):
    inds = mask.flatten()
    inds[ 0] = 0
    inds[-1] = 0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def run_length_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b


    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths


def run_length_decode(rel, H, W, fill_value=255):
    mask = np.zeros((H*W),np.uint8)
    rel  = np.array([int(s) for s in rel.split(' ')]).reshape(-1,2)
    for r in rel:
        start = r[0]
        end   = start +r[1]
        mask[start:end]=fill_value
    mask = mask.reshape(H,W)
    return mask



# check #################################################################
def run_check_rle():

    #mask_file = '/root/share/project/kaggle/science2018/data/stage1_train/d8607b21411c9c8ab532faaeba15f8818a92025897950f94ee4da4f74f53660a/masks/' \
    #            + '93a160042ae86c42f7645e608488080032c19589f0b02512b42a02e7f73fc426.png'
    mask_file = '/root/share/project/kaggle/science2018/data/stage1_train/d8607b21411c9c8ab532faaeba15f8818a92025897950f94ee4da4f74f53660a/masks/' \
                + '93a160042ae86c42f7645e608488080032c19589f0b02512b42a02e7f73fc426.png'
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    rle  = run_length_encode(mask>128)

    print(rle)





# main #################################################################

if __name__ == '__main__':

    run_check_rle()


    print('\nsucess!')
