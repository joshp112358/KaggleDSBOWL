import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'


from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.science_dataset import *
from dataset.sampler import *


# -------------------------------------------------------------------------------------
from net.model.unet1 import UNet256_3x3 as Net


def debug_show_probs(tensors, masks, probs, wait=1, is_save=True, dir=None):
    if is_save == True:
        os.makedirs(dir, exist_ok=True)

    batch_size = len(tensors)
    #print(batch_size)

    images = tensors.data.cpu().numpy()
    masks  = masks.data.cpu().numpy()
    probs  = probs.data.cpu().numpy()
    for m in range(batch_size):
        image = images[m].transpose((1,2,0))*255
        mask  = masks[m]*255
        prob  = probs[m]*255

        image = image.astype(np.uint8)
        mask  = mask.astype(np.uint8)[:, :, np.newaxis]*np.array([1,1,1],np.uint8)
        prob  = prob.astype(np.uint8)[:, :, np.newaxis]*np.array([1,1,1],np.uint8)
        all = np.hstack((image,mask,prob))

        if is_save == True:
            cv2.imwrite(out_dir +'/train/iterations/%08d-%03d.png'%(i,m),all)

        # image_show('image',image)
        # image_show('mask',mask)
        # image_show('prob',prob)
        image_show('all',all)
        cv2.waitKey(wait)


def train_augment(image,mask,index):
    image, mask = fix_resize_transform2(image, mask, SCIENCE_WIDTH, SCIENCE_HEIGHT)
    tensor = image.transpose((2,0,1))
    tensor = torch.from_numpy(tensor).float().div(255)
    mask   = torch.from_numpy(mask).float().div(255)
    return tensor,mask,index



def valid_augment(image,mask,index):
    image, mask = fix_resize_transform2(image, mask, SCIENCE_WIDTH, SCIENCE_HEIGHT)

    tensor = image.transpose((2,0,1))
    tensor = torch.from_numpy(tensor).float().div(255)
    mask   = torch.from_numpy(mask).float().div(255)
    return tensor,mask,index


#--------------------------------------------------------------
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for i, (tensors, masks, indices) in enumerate(test_loader, 0):
        tensors = Variable(tensors,volatile=True).cuda()
        masks   = Variable(masks).cuda()

        logits = data_parallel(net, tensors)
        loss   = BCELoss2d()(logits, masks)


        batch_size = len(indices)
        test_acc  += 0  #batch_size*acc[0][0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc



#--------------------------------------------------------------
def run_train():

    out_dir  = RESULTS_DIR + '/unet-00'
    initial_checkpoint = \
       None  #RESULTS_DIR + '/checkpoint/00021000_model.pth'


    pretrain_file = None #imagenet pretrain

    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (3,256,256), num_classes=1).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    elif pretrain_file is not None:
        log.write('\tpretrained_file = %s\n' % pretrain_file)
        #load_pretrain_file(net, pretrain_file)


    log.write('%s\n\n'%(type(net)))
    log.write('\n')


    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 32  ##NUM_CUDA_DEVICES*512 #256//iter_accum #512 #2*288//iter_accum

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 50
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,100))#1*1000

    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = None
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(
                                'train_ids_remove_error_669', mode='train',
                                transform = train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = collate)

    valid_dataset = ScienceDataset(
                                'train_ids_remove_error_669', mode='train', #<todo> make validation set
                                 transform = valid_augment)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = collate)

    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    #log.write('\n')

    if 0: #<debug>
        for tensors, masks, indices in train_loader:
            batch_size = len(indices)
            print(batch_size)

            images = tensors.cpu().numpy()
            masks  = masks.cpu().numpy()
            for i in range(batch_size):
                image = images[i].transpose((1,2,0))*255
                mask  = masks[i]*255


                image_show('image',image)
                image_show('mask',mask)
                cv2.waitKey(0)



    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' waves_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate   iter_k   epoch  num_m | valid_loss/acc | train_loss/acc | batch_loss/acc |  time    \n')
    log.write('--------------------------------------------------------------------------------------------\n')


    train_loss  = 0.0
    train_acc   = 0.0
    valid_loss  = 0.0
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0


    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0

        net.train()
        optimizer.zero_grad()
        for tensors, masks, indices in train_loader:
            batch_size = len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid==0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k  %6.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s \n' % \
                        (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, \
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))



            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum


            # one iteration update  -------------
            tensors = Variable(tensors).cuda()
            masks   = Variable(masks).cuda()
            logits  = data_parallel(net, tensors)
            probs   = F.sigmoid(logits).squeeze(1)

            loss = BCELoss2d()(logits, masks)
            acc  = 0 #<todo>



            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


            if 0: #<debug>
                debug_show_probs(tensors, masks, probs,
                                 wait=1, is_save=False, dir=out_dir +'/train/iterations')



            # print statistics  ------------
            batch_acc  = 0#acc[0][0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.
                sum_train_acc  = 0.
                sum = 0

            print('\r%0.4f  %5.1f k  %6.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s  %d,%d, %s' % \
                    (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                     time_to_str((timer() - start)/60) ,i,j, str(tensors.size())), end='',flush=True)
            j=j+1
        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1:
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#
#