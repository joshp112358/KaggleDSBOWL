from common import *
from utility.file import *
from utility.draw import *



## run ###################################################################################3
def run_count_image():
    #split = 'test_all_ids_65'
    split = 'train_all_ids_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        image_files =   glob.glob(DATA_DIR + '/' + id + '/images/*.png')
        print(i, len(image_files))

        assert(len(image_files)==1)
        image_file=image_files[0]
        image =cv2.imread(image_file)

        image_show('image',image)
        cv2.waitKey(100)



def run_make_one_mask():

    split = 'train_all_ids_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        image_files =   glob.glob(DATA_DIR + '/' + id + '/images/*.png')
        print(i, len(image_files))

        assert(len(image_files)==1)
        image_file=image_files[0]
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        H,W,C = image.shape
        one_mask = np.zeros((H,W), dtype=bool)
        mask_files =   glob.glob(DATA_DIR + '/' + id + '/masks/*.png')
        for mask_file in mask_files:
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            one_mask = one_mask |(mask>128)


            #image_show('mask',mask)
            #image_show('one_mask',(one_mask*255).astype(np.uint8))
            #cv2.waitKey(100)

        one_mask = (one_mask*255).astype(np.uint8)
        cv2.imwrite(DATA_DIR + '/' + id + '/one_mask.png', one_mask)

        image_show('one_mask',one_mask)
        cv2.waitKey(10)


# main #################################################################

if __name__ == '__main__':


    run_make_one_mask()
    print('\nsucess!')




