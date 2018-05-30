from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *

SCIENCE_WIDTH =256
SCIENCE_HEIGHT=256





#
def collate(batch):
    batch_size = len(batch)
    num     = len(batch[0])
    indices = [batch[b][num-1]for b in range(batch_size)]
    tensors = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    if batch[0][1] is None:
        masks = None
    else:
        masks = torch.stack([batch[b][1]for b in range(batch_size)], 0)
    return [tensors,masks,indices]


# #data iterator ----------------------------------------------------------------
class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()

        self.split = split
        self.transform = transform
        self.mode = mode

        #read split
        ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

        #save
        self.ids = ids

        #print
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('\tnum_ids = %d'%(len(self.ids)))
        print('')


    def __getitem__(self, index):
        id   = self.ids[index]
        name = id.split('/')[-1]
        image_file = DATA_DIR + '/' + id + '/images/' + name +'.png'
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            mask_file =  DATA_DIR + '/' + id + '/one_mask.png'
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                return self.transform(image,mask,index)
            else:
                return image, mask, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image,index)
            else:
                return image, index



    def __len__(self):
        return len(self.ids)




## check ## ----------------------------------------------------------

def run_check_dataset():

    def augment(image,mask,index):
        image,mask = fix_resize_transform2(image, mask, SCIENCE_WIDTH, SCIENCE_HEIGHT)
        return image,mask,index

    dataset = ScienceDataset(
        'train_ids_remove_error_669', mode='train',
        transform = augment,
    )
    #sampler = SequentialSampler(dataset)
    sampler = RandomSampler(dataset)


    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        image,mask,index = dataset[n]

        image_show('image',image)
        image_show('mask',mask)
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()

    print( 'sucess!')
