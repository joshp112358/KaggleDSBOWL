import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.science_dataset import *
from dataset.sampler import *
from dataset.mask import *


# -------------------------------------------------------------------------------------
from train_unet import *




def submit_augment(image, index):

    image1 = fix_resize_transform(image, SCIENCE_WIDTH, SCIENCE_HEIGHT)
    tensor = image1.transpose((2,0,1))
    tensor = torch.from_numpy(tensor).float().div(255)

    return tensor,image,index



def submit_collate(batch):
    batch_size = len(batch)
    num     = len(batch[0])
    indices = [batch[b][num-1]for b in range(batch_size)]
    tensors = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    sizes   = [batch[b][1]for b in range(batch_size)]
    return [tensors,sizes,indices]

#--------------------------------------------------------------

def do_submit():


    out_dir  = RESULTS_DIR + '/unet-00'
    initial_checkpoint = \
       RESULTS_DIR + '/unet-00/checkpoint/00002500_model.pth'

    #output
    csv_file    = out_dir +'/submit/submission-2.csv'


    ## setup -----------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n')

    net = Net(in_shape = (3,256,256), num_classes=1).cuda()
    net.load_state_dict(torch.load(initial_checkpoint))
    net.eval()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n\n'%(type(net)))


    ## dataset ---------------------------------
    log.write('** dataset setting **\n')
    test_dataset = ScienceDataset(
                                #'train1_ids_remove_error_669', mode='test',
                                'test1_ids_all_65',  mode='test',
                                 transform = submit_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = submit_collate)
    test_num  = len(test_loader.dataset)


    ## start submission here ####################################################################
    start = timer()

    predicts = [];
    cvs_ImageId = [];
    cvs_EncodedPixels = [];
    n = 0
    for tensors, images, indices in test_loader:
        batch_size = len(indices)

        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(n, test_num, 100*n/test_num,
                         (timer() - start) / 60), end='',flush=True)
        time.sleep(0.01)

        # forward
        tensors = Variable(tensors,volatile=True).cuda(async=True)
        logits  = data_parallel(net, tensors)
        probs   = F.sigmoid(logits).squeeze(1)


        #save
        probs  = probs.data.cpu().numpy()
        for m in range(batch_size):
            id = test_dataset.ids[indices[m]].split('/')[-1]
            image = images[m]
            h,w = image.shape[:2]

            prob = probs[m]*255
            prob = prob.astype(np.uint8)
            prob = cv2.resize(prob,(w,h))
            predict  = cv2.threshold(prob, 75, 255, cv2.THRESH_BINARY)[1]#128
            predicts.append(predict)


            #temporay solution -------------------------
            #https://www.kaggle.com/kmader/nuclei-overview-to-submission
            label = skimage.morphology.label(predict>128)

            num = label.max()+1
            for m in range(1, num):
                rle = run_length_encode(label==m)
                cvs_ImageId.append(id)
                cvs_EncodedPixels.append(rle)


            #plt.imshow(label)
            #plt.show()
            #temporay solution -------------------------


            #<debug>
            prob    = prob.astype(np.uint8)[:, :, np.newaxis]*np.array([1,1,1],np.uint8)
            predict = predict.astype(np.uint8)[:, :, np.newaxis]*np.array([1,1,1],np.uint8)
            label_overlay = label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255
            label_overlay = label_overlay.astype(np.uint8)

            all = np.hstack((image,prob,predict))
            image_show('all',all)
            image_show('label_overlay',label_overlay)
            cv2.waitKey(1)

        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)


    ## submission csv  ----------------------------
    df = pd.DataFrame({ 'ImageId' : cvs_ImageId , 'EncodedPixels' : cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_submit()


    print('\nsucess!')