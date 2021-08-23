## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.
# This is for Basic transformation + both l2 distances
import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
from setup_imagenet_hash2 import ImageNet, ImageNet_HashModel
from setup_face_hash2 import Face, Face_HashModel # 2 for rgb version 
#from setup_maladv_hash import MalAdv, MalAdv_HashModel
from l2_attack_black_rgb import BlackBoxL2

from PIL import Image
import imagehash
import robusthash
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from numpy import pi
from numpy import sin
from numpy import zeros
from scipy import signal
import matplotlib.pylab as pylab
import scipy.fftpack
from skimage import transform
import cv2
from skimage import exposure
from lpips_tensorflow.lpips_tf import lpips

def show(img, name="output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save(name, img)
    fig = np.around((img+0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    gray_inputs = []
    for i in range(samples):
        if targeted:
            if inception:
                # for inception, randomly choose 10 target classes
                seq = np.random.choice(range(1, 1001), 10)
                # seq = [580] # grand piano
            else:
                # for CIFAR and MNIST, generate all target classes
                seq = range(data.test_labels.shape[1])

            # print ('image label:', np.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                # targets.append(np.eye(data.test_labels.shape[1])[j])
                # labels.append(data.test_labels[start + i])
                true_ids.append(start + i)
        else:
            inputs.append(data.test_data[start + i])
            # targets.append(data.test_labels[start + i])
            # labels.append(data.test_labels[start + i])
            true_ids.append(start + i)
            gray_inputs.append(data.test_data_gray[start + i])
            # dct_compression(data.test_data_gray[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)
    gray_inputs = np.array(gray_inputs)

    return inputs, targets, labels, true_ids, gray_inputs

def gen_image(arr):
    # two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    # img = Image.fromarray(np.uint8(arr * 255))
    # img = Image.fromarray(two_d)

    fig = np.around((arr+0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img



def main(args):
    # config = tf.ConfigProto(device_count={"CPU": 2}, # limit to num_cpu_core CPU usage
    #             inter_op_parallelism_threads = 1, 
    #             intra_op_parallelism_threads = 4,
    #             log_device_placement=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']  # "0,1,2,3".
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

    #Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    #Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        use_log = not args['use_zvalue']
        is_inception = args['dataset'] == "imagenet"
        # load network
        print('Loading model', args['dataset'])
        if args['dataset'] == "mnist":
            data, model = MNIST(), MNIST_HashModel()

            # data, model =  MNIST(), MNISTModel("models/mnist-distilled-100", sess, use_log)
        elif args['dataset'] == "cifar10":
            data, model = CIFAR(), CIFAR_HashModel()
        #     # data, model = CIFAR(), CIFARModel("models/cifar-distilled-100", sess, use_log)
        elif args['dataset'] == "imagenet":
            data, model = ImageNet(), ImageNet_HashModel(args['hash'], args['bits'], args['factor'])
        # elif args['dataset'] == "maladv":
        #     data, model = MalAdv(), MalAdv_HashModel()
        elif args['dataset'] == "face":
            data, model = Face(), Face_HashModel(args['hash'], args['bits'], args['factor'])
     
        if args['numimg'] == 0:
            args['numimg'] = len(data.test_labels) - args['firstimg']
        print('Using', args['numimg'], 'test images')
        # load attack module
        if args['attack'] == "white":
            # batch size 1, optimize on 1 image at a time, rather than optimizing images jointly
            # attack = CarliniL2(sess, model, batch_size=1, max_iterations=args['maxiter'],
            #                    print_every=args['print_every'],
            #                    early_stop_iters=args['early_stop_iters'], confidence=0, learning_rate=args['lr'],
            #                    initial_const=args['init_const'],
            #                    binary_search_steps=args['binary_steps'], targeted=not args['untargeted'],
            #                    use_log=use_log,
            #                    adam_beta1=args['adam_beta1'], adam_beta2=args['adam_beta2'])
            print('white')
        else:
            # batch size 128, optimize on 128 coordinates of a single image
            attack = BlackBoxL2(sess, model, batch_size=args['batch'], max_iterations=args['maxiter'],
                                print_every=args['print_every'],
                                early_stop_iters=args['early_stop_iters'], confidence=0, learning_rate=args['lr'],
                                initial_const=args['init_const'],
                                binary_search_steps=args['binary_steps'], targeted=not args['untargeted'],
                                use_log=use_log, use_tanh=args['use_tanh'],
                                use_resize=args['use_resize'], adam_beta1=args['adam_beta1'],
                                adam_beta2=args['adam_beta2'], reset_adam_after_found=args['reset_adam'],
                                solver=args['solver'], save_ckpts=args['save_ckpts'], load_checkpoint=args['load_ckpt'],
                                start_iter=args['start_iter'],
                                init_size=args['init_size'], use_importance=not args['uniform'], method=args['method'], dct=args['dct'], dist_metrics=args['dist_metrics'], htype=args["htype"])

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        print('Generate data')

        all_inputs, all_targets, all_labels, all_true_ids, all_gray_inputs = generate_data(data, samples=args['numimg'],
                                                                          targeted=not args['untargeted'],
                                                                          start=args['firstimg'],
                                                                          inception=is_inception)
        # all_gray_inputs_tfomr1, all_gray_inputs_tform2 = generate_transformation(all_gray_inputs)
        print('Done...')
        os.system("mkdir -p {}/{}".format(args['save'], args['dataset']))
        img_no = args["start_idx"]
        total_success = 0
        # l2_total = 0.0
        l2_total = 0.0
        pdistance1_total = 0.0
        hash_total = 0.0
        hash_total2 = 0
      
        l2_total2 = 0
        pdistance_total = 0


        # print('testing for phash differences for %s dataset' % args['dataset'])
        differences = 0
        total = 0

        total_success_iterations = 0
        total_iterations = 0

        for i in range(args['start_idx'], min((args['start_idx'] + 50), all_true_ids.size)):
        # for i in start_idxs:
            print('for image id ', all_true_ids[i])
            inputs = all_inputs[i:i + 1]
            gray_inputs = all_gray_inputs[i:i+1]
            print('each rgb inputs shape ', inputs.shape)
            print('each gray inputs shape ', gray_inputs.shape)
            # if len(gray_inputs.shape) == 4:
            #     gray_inputs = gray_inputs[0]
            best_modifier_path = './{}/best_modifier_img{}.npy'.format(args['save_ckpts'], i)

            img_no += 1
        
            timestart = time.time()
            adv, const, L3, adv_current, first_iteration = attack.attack_batch(gray_inputs, inputs, i, best_modifier_path)
            timeend = time.time()
            if type(const) is list:
                const = const[0]
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            if len(adv_current.shape) == 3:
                adv_current = adv_current.reshape((1,)+ adv_current.shape)
           
            # l2 distances
            l2_distortion_direct = np.sum((adv - inputs) ** 2) ** 0.5
            print('l2_distortion between inputs and adv ', l2_distortion_direct)
            l2_distortion_current = np.sum((adv_current - inputs) ** 2) ** 0.5
            print('l2_distortion between inputs and adv_current ', l2_distortion_current)
            if len(inputs.shape) == 4:
                a,b,c = inputs[0].shape
                print("normalized a,b,c", a, b, c)
                l2_distortion_normalized = l2_distortion_direct / (a*b*c)**0.5
                l2_distortion_current_normalized = l2_distortion_current / (a*b*c)**0.5
                print("normalized l2 between inputs and adv ", l2_distortion_normalized)
                print("normalized l2 between inputs and adv_current ", l2_distortion_current_normalized)

            # if len(gray_inputs.shape) == 4:
            #     stacked_inputs = inputs[0]
            #     stacked_adv = adv[0]
            #     stacked_adv_current = adv_current[0]


            image0_ph = tf.placeholder(tf.float32)
            image1_ph = tf.placeholder(tf.float32)
            distance_t = lpips(image0_ph, image1_ph, model='net-lin', net='alex')
           
            # stacked_inputs =  np.asarray(np.dstack((stacked_inputs, stacked_inputs, stacked_inputs)))
            # stacked_adv =  np.asarray(np.dstack((stacked_adv, stacked_adv, stacked_adv)))
            # stacked_adv_current =  np.asarray(np.dstack((stacked_adv_current, stacked_adv_current, stacked_adv_current)))
            # if len(stacked_inputs.shape) == 3:
            #     stacked_inputs = stacked_inputs.reshape((1,) + stacked_inputs.shape)
            #     stacked_adv = stacked_adv.reshape((1,) + stacked_adv.shape)
            #     stacked_adv_current = stacked_adv_current.reshape((1,) + stacked_adv_current.shape)
        
            # print(stacked_adv.shape)
            # print(stacked_adv_current.shape)
            # perceptual distances
            with tf.Session(config=config) as session:
                distance1 = session.run(distance_t, feed_dict={image0_ph: (inputs+0.5), image1_ph: (adv+0.5)})
                distance2 = session.run(distance_t, feed_dict={image0_ph: (inputs+0.5), image1_ph: (adv_current+0.5)})

            # distance1_normalized = distance1[0] / (a*b*c)**0.5
            # distance2_normalized = distance2[0] / (a*b*c)**0.5
            success = False

            print("perceptual metrics distance between adv and img", distance1[0])
            print("perceptual metrics distance between adv_current and img", distance2[0])
            # print("perceptual metrics distance between adv and img", distance1_normalized)
            # print("perceptual metrics distance between adv_current and img", distance2_normalized)

            inputs_img = gen_image(inputs)
            adv_img = gen_image(adv)
           

            if args["htype"] == "phash":
                hash_differences = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_img, args['bits'], args['factor'])
                print('perceptual hash difference is ', hash_differences)
            elif args["htype"] == "blockhash":
                if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                    im_original = inputs_img.convert('RGB')
                    im_adver = adv_img.convert('RGB')
                else:
                    im_original = inputs_img
                    im_adver = adv_img
                hash_differences = sum(1 for i, j in zip(robusthash.blockhash(im_original), robusthash.blockhash(im_adver)) if i != j)
                print('robust hash differences', hash_differences)
           
            # inputs_arr = np.asarray(inputs_img) / 255.0 
            # adv_arr = np.asarray(adv_img) / 255.0 
            # l2_distortion = np.sum((inputs_arr - adv_arr) ** 2) ** 0.5

            if L3 == True: 
                # if args["htype"] == "phash":
                #     hash_differences = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_img, args['bits'], args['factor'])
                #     print('perceptual hash difference is ', hash_differences)
                # elif args["htype"] == "blockhash":
                #     if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                #         im_original = inputs_img.convert('RGB')
                #         im_adver = adv_img.convert('RGB')
                #     hash_differences = sum(1 for i, j in zip(robusthash.blockhash_even(im_original), robusthash.blockhash_even(im_adver)) if i != j)
                #     print('robust hash differences', hash_differences)


                if args['untargeted']:
                    if hash_differences >= args['hash']:
                        #print('hash difference threshold ', args['hash'])
                        success = True
           
                # if l2_distortion_direct > 20.0:
                #     success = False
                
                # if len(inputs.shape) == 4:
                #     a,b,c = inputs[0].shape
                #     l2_distortion_normalized = l2_distortion_direct / (a*b*c)**0.5
                if success:
                    total_success += 1
                    l2_total += l2_distortion_direct
                    pdistance1_total += distance1[0]
                    hash_total += hash_differences
                    hash_total2 += hash_differences
                    l2_total2 += l2_distortion_direct
                    pdistance_total += distance1[0]
                    total_success_iterations += first_iteration
                    total_iterations += first_iteration
                    suffix = "id{}_differ{}_{}_l2normalized{:.2f}_pdistance{}".format(all_true_ids[i], hash_differences, success, l2_distortion_direct, distance1[0])
                    print("Saving to", suffix)
    
                    show(inputs, "{}/{}/{}_original_{}.png".format(args['save'], args['dataset'], img_no,suffix))
                    show(adv, "{}/{}/{}_adversarial_{}.png".format(args['save'], args['dataset'], img_no, suffix))
                    # show(adv - inputs, "{}/{}/{}_diff_{}.png".format(args['save'], args['dataset'], img_no, suffix))
            
                    # print("[STATS][L1] total = {}, id = {}, time = {:.3f}, success = {}, const = {:.6f}, hash_avg={:.5f}, distortion = {:.5f}, success_rate = {:.3f}, l2_normalized_avg={:.5f}, p_normalized_avg={}"
                    # .format(img_no, all_true_ids[i], timeend - timestart, success, const, 0 if total_success == 0 else hash_total / total_success, l2_distortion_normalized, total_success / float(img_no), 0 if total_success == 0 else l2_normalized_total / total_success, 0 if total_success == 0 else distance1_normalized_total/ total_success))
                    print("[STATS][L1] total = {}, id = {}, time = {:.3f}, success = {}, const = {:.6f}, hash_avg={:.5f}, distortion = {:.5f}, success_rate = {:.3f}, l2_avg={:.5f}, p_avg={}, iteration_avg={}"
                    .format(img_no, all_true_ids[i], timeend - timestart, success, const, 0 if total_success == 0 else hash_total / total_success, l2_distortion_normalized, total_success / float(img_no), 0 if total_success == 0 else l2_total / total_success, 0 if total_success == 0 else pdistance1_total/ total_success
                    , 0 if total_success == 0 else total_success_iterations/ total_success))                    
                    
                    sys.stdout.flush()
            else:
                
                adv_current_img = gen_image(adv_current)
                print("unsuccessful")
                if args["htype"] == "phash":
                    hash_differences_current = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_current_img, args['bits'], args['factor'])
                    print('perceptual hash difference is ', hash_differences_current)
                elif args["htype"] == "blockhash":
                    if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                        im_original = inputs_img.convert('RGB')
                        im_adver_current = adv_current_img.convert('RGB')
                    else:
                        im_original = inputs_img
                        im_adver_current = adv_current_img
                    hash_differences_current = sum(1 for i, j in zip(robusthash.blockhash(im_original), robusthash.blockhash(im_adver_current)) if i != j)
                total_iterations += first_iteration
                hash_total2 += hash_differences_current
                l2_total2 += l2_distortion_current
                pdistance_total +=  distance2[0]
                print("Failed attacks!")
                suffix_current = "l2{:.2f}_pdist{}_diff{}_success={}_time{}".format(l2_distortion_current_normalized, distance2[0], hash_differences_current, success, timeend - timestart)
                show(adv_current, "{}/{}/id{}_adv_current_{}.png".format(args['save'], args['dataset'], i, suffix_current))
                print('saving for failed attack current', suffix_current)
                sys.stdout.flush()

            print("overall average hash ", hash_total2 / (img_no - args["start_idx"]))
            print("overall average l2 ", l2_total2 /(img_no-args["start_idx"]))
            print("overal average perceptual distance ", pdistance_total / (img_no-args["start_idx"]))
            print("overal average iterations ", total_iterations / (img_no-args["start_idx"]))
            sys.stdout.flush()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet", "maladv", "face", "random"], default="imagenet")
    parser.add_argument("-s", "--save", default="./saved_results")
    parser.add_argument("-a", "--attack", choices=["white", "black"], default="black")
    parser.add_argument("-n", "--numimg", type=int, default=0, help="number of test images to attack")
    parser.add_argument("-m", "--maxiter", type=int, default=0, help="set 0 to use default value")
    parser.add_argument("-p", "--print_every", type=int, default=100, help="print objs every PRINT_EVERY iterations")
    parser.add_argument("-o", "--early_stop_iters", type=int, default=0,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("-f", "--firstimg", type=int, default=0)
    parser.add_argument("-b", "--binary_steps", type=int, default=0)
    parser.add_argument("-c", "--init_const", type=float, default=0.0)
    parser.add_argument("-z", "--use_zvalue", action='store_true')
    parser.add_argument("-u", "--untargeted", action='store_true')
    parser.add_argument("-r", "--reset_adam", action='store_true', help="reset adam after an initial solution is found")
    parser.add_argument("--use_resize", action='store_true', help="resize image (only works on imagenet!)")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1216)
    parser.add_argument("--solver", choices=["adam", "adam2", "adam2_newton", "newton", "adam_newton", "fake_zero"], default="adam")
    parser.add_argument("--save_ckpts", default="", help="path to save checkpoint file")
    parser.add_argument("--load_ckpt", default="", help="path to numpy checkpoint file")
    parser.add_argument("--start_iter", default=0, type=int,
                        help="iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--init_size", default=32, type=int, help="starting with this size when --use_resize")
    parser.add_argument("--uniform", action='store_true', help="disable importance sampling")
    parser.add_argument("--method", "--transform_method",  default='linear')
    parser.add_argument("--gpu", "--gpu_machine", default="0")
    parser.add_argument("--hash", "--hashbits", type=int, default=6)
    parser.add_argument("--batch", "--batchsize", type=int, default=128)
    parser.add_argument("--start_idx", "--start image index", type=int, default=0)
    parser.add_argument("--dct", "--if using dct compression", action='store_true')
    # parser.add_argument("--num_rand_vec", "--random number of vectors like batch", type=int, default=1)
    parser.add_argument("--lr", "--learning rate", type=float, default=0.1)
    parser.add_argument("--transform", "--basic transormation", default="centrol_crop")
    parser.add_argument('--dist_metrics', "--distance metrics to use", choices=["l2dist", "pdist"], default="l2dist")
    parser.add_argument("--bits", "--hash_string_length", type=int, default=8)
    parser.add_argument("--factor", "--hash_string_factor", type=int, default=4)
    parser.add_argument("--maximize", "--if_plus_or_minus", choices=["plus", "minus"], default="minus")
    parser.add_argument("-ht", "--htype", choices=["phash", "blockhash"], default="phash")
    args = vars(parser.parse_args())
    # add some additional parameters
    # learning rate
    #args['lr'] = 1e-2
    args['inception'] = False
    args['use_tanh'] = True
    # args['use_resize'] = False
    if args['maxiter'] == 0:
        if args['attack'] == "white":
            args['maxiter'] = 1000
        else:
            if args['dataset'] == "imagenet":
                if args['untargeted']:
                    # for imagenet resize
                    args['maxiter'] = 2000
                else:
                    args['maxiter'] = 50000
            elif args['dataset'] == "mnist":
                args['maxiter'] = 2000
            elif args['dataset'] == "face":
                args['maxiter'] = 2000
    if args['init_const'] == 0.0:
        if args['binary_steps'] != 0:
            args['init_const'] = 100
        else:
            args['init_const'] = 1
    if args['binary_steps'] == 0:
        args['binary_steps'] = 1
    # set up some parameters based on datasets
    # if args['dataset'] == "imagenet":
    #     args['inception'] = True
    #     args['lr'] = 1e-3
    #     # args['use_resize'] = True
    #     # args['save_ckpts'] = True
    # if args['dataset'] == "maladv" or args['dataset'] == "face":
    #     args['lr'] = 2e-3
    # for mnist, using tanh causes gradient to vanish
    if args['dataset'] == "mnist":
        args['use_tanh'] = True
    # when init_const is not specified, use a reasonable default
    # if args['init_const'] == 0.0:
    #     if args['binary_search']:
    #         args['init_const'] = 0.01
    #     else:
    #         args['init_const'] = 0.5
    # setup random seed
    args['load_ckpt'] = './{}/best_modifier_img6.npy'.format(args['save_ckpts'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    print(args)
    main(args)
