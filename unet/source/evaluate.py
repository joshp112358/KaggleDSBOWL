import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.science_dataset import *
from dataset.sampler import *


# --------------------------------------------------------------
from train_unet import *


eval_augment = valid_augment
#--------------------------------------------------------------
def run_evaluate():

    out_dir  = RESULTS_DIR + '/unet-00'
    initial_checkpoint = \
       RESULTS_DIR + '/unet-00/checkpoint/00002500_model.pth'


    pretrain_file=\
        None 


    ## setup  ---------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (3,256,256), num_classes=1).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    elif pretrain_file is not None:  #pretrain
        log.write('\tpretrain_file    = %s\n' % pretrain_file)
        #load_pretrain_file(net, pretrain_file)


    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
                                'train1_ids_remove_error_669', mode='train',
                                transform = eval_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = collate)


    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')




    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    net.eval()

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for i, (tensors, masks, indices) in enumerate(test_loader, 0):

        tensors = Variable(tensors,volatile=True).cuda()
        masks   = Variable(masks).cuda()
        logits  = data_parallel(net, tensors)
        probs   = F.sigmoid(logits).squeeze(1)

        loss = BCELoss2d()(logits, masks)
        acc  = 0 #<todo>

        if 1: #<debug>
            debug_show_probs(tensors, masks, probs,
                             wait=1, is_save=False, dir=out_dir +'/eval/iterations')



        batch_size = len(indices)
        test_acc  += 0 #batch_size*acc[0][0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num


    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_acc  = %0.5f\n'%(test_acc))
    log.write('test_loss = %0.5f\n'%(test_loss))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_evaluate()


    print('\nsucess!')