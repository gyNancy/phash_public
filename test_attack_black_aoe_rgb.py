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

from setup_mnist_hash import MNIST_HashModel
from setup_mnist_hash import MNIST
from setup_imagenet_hash2 import ImageNet, ImageNet_HashModel
from setup_cifar_hash import CIFAR, CIFAR_HashModel
#from setup_maladv_hash import MalAdv, MalAdv_HashModel
from setup_face_hash2 import Face, Face_HashModel
from setup_random_hash import Random, Random_HashModel
# from l2_attack import CarliniL2
from l2_attack_black_eot_rgb import BlackBoxL2

from PIL import Image
import imagehash
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
import robusthash
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

# def transformation(gray_inputs):
#     tform1 = transform.AffineTransform(scale=(0.98,0.98),rotation=0.02,translation=(6,6))
#     #transformed_img = transform.warp(inputs, tform1)
#     transformed_gray_inputs = transform.warp(gray_inputs, tform1)
def generate_aot(gray_inputs):
    height = gray_inputs.shape[0]
    width = gray_inputs.shape[1]
    gray_inputs1 = resize(gray_inputs, (height, int(width * 1.1)),anti_aliasing=True)

    clip = int((width * 1.1 - width)/2)
    gray_inputs1 = np.array(gray_inputs1[:, clip:clip+width])

    return gray_inputs1

def generate_transformation(all_gray_inputs, i):
    height, width = all_gray_inputs.shape[0], all_gray_inputs.shape[1]
    # tform1 = transform.AffineTransform(scale=(0.98,0.98),rotation=0.02,translation=(6,6))
    # if dataset == "imagenet":
    #     tform2 = transform.AffineTransform(rotation=0.05,translation=(15,15)) # 0.05
    #     imgs_tform2 = transform.warp(all_gray_inputs, tform2)
    #     # imgs_tform1 = imgs_tform1[10:289, 10:289]
    #     rotation_array= imgs_tform2[5:285, 5:285] # ~5% central cropping

    #     cropped_array = all_gray_inputs[20:279, 20:279] # 15%
        
    # elif dataset == "face":
    #tform2 = transform.AffineTransform(rotation=0.05,translation=(int(height * 0.05), int(width*0.05))) # 0.05
    tform2 = transform.AffineTransform(rotation=0.02) # 0.05
    imgs_tform2 = transform.warp(all_gray_inputs, tform2)
    #rotation_array= imgs_tform2[int(height*0.1/ 2): int(height - (height*0.1)/ 2), int(height*0.1/ 2):int(width - (width*0.1)/ 2)] #0.05
    rotation_array= imgs_tform2[int ((height*0.05)/ 2): int(height - (height*0.05)/ 2), int ((width*0.05)/ 2):int(width - (width*0.05)/ 2), :] #0.05
    cropped_array = all_gray_inputs[0: int(height - (height*0.05)/ 2), 0:int(width - (width*0.05)/ 2), :] # 10%
    aot_array = generate_aot(all_gray_inputs)
# cropped_array7 = all_gray_inputs[30:269, 30:269] 
    # v_min, v_max = np.percentile(all_gray_inputs, (5, 95))
    # contrast8= exposure.rescale_intensity(all_gray_inputs, in_range=(v_min, v_max))
    # v_min1, v_max1 = np.percentile(all_gray_inputs, (2, 98))
    # contrast9= exposure.rescale_intensity(all_gray_inputs, in_range=(v_min1, v_max1))
    # for random people
    # cropped_array5 = all_gray_inputs[10:290, 10:440] 
    # cropped_array6 = all_gray_inputs[20:280, 20:430] 

    def resize_array(img_array, height, width):
        # resized_array = cv2.resize(img_array, (299,299), interpolation=cv2.INTER_AREA)
        resized_array = resize(img_array, (height,width),anti_aliasing=True)
        resized_array = np.resize(resized_array, (resized_array.shape[0], resized_array.shape[1], 3))

        return resized_array

    # resized1 = resize(imgs_tform1)
    resized2 = resize_array(rotation_array, height, width)
    resized6 = resize_array(cropped_array, height, width)
    # print("aot shape =======", aot_array.shape)
    # resized7 = resize_array(aot_array, height, width)
    # show(resized2, "resized2_{}.png".format(i))
    # show(resized6, "resized6_{}.png".format(i))
    # show(aot_array, "aot_{}.png".format(i))
    # print('resized2 non zero', resized2[-1, -1, -1] != 0)
    # print('resized6 non zero', resized6[-1, -1, -1] != 0)
    # print('aot non zero', aot_array[-1, -1, -1] != 0)
    # resized7 = resize(cropped_array7)
    # resized8 = resize(contrast8)
    # resized9 = resize(contrast9)
   

    #return resized1, resized2, resized3, resized4, resized5, resized6, resized7, resized8, resized9
    return resized2, resized6, aot_array

def basic_transformation(gray_inputs, basic_trans):
    if basic_trans == "centrol_crop":
        cropped_array = gray_inputs[20:279, 20:279]  #20%
        resized_array = cv2.resize(cropped_array, (299,299), interpolation=cv2.INTER_AREA) # inter_nearest, inter_linear, inter_area, inter_cubic, inter_lanczos4
      
    elif basic_trans == "rotation":
        tform1 = transform.AffineTransform(rotation=0.03) #30%
        rotated_array = transform.warp(gray_inputs, tform1)
        cropped_array = rotated_array[10:289, 10:289]
        print('if cutting all black edges ', cropped_array[-1,-1,-1] != 0)
        resized_array = cv2.resize(cropped_array, (299,299), interpolation=cv2.INTER_AREA)

    if len(resized_array.shape) == 2:
        resized_array = np.resize(resized_array, (resized_array.shape[0], resized_array.shape[1], 1))
    return resized_array


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
        elif args['dataset'] == "random":
            data, model = Random(), Random_HashModel(args['hash'], args['bits'], args['factor'])
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
        l2_total = 0.0
        pdistance1_total = 0.0
        hash_total = 0.0
        hash_total2 = 0
        # print('testing for phash differences for %s dataset' % args['dataset'])
        l2_total2 = 0
        pdistance_total = 0
        differences = 0
        total = 0

        total_success_iterations = 0
        total_iterations = 0
        total_time = 0


        total_transformed_hash1 = []
        total_transformed_hash2 = []
        total_transformed_hash3 = []
        total_transforms = []
    
        for i in range(args['start_idx'], min((args['start_idx'] + 50), all_true_ids.size)):
        #for i in start_idxs:
            print('for image id ', i)
            inputs = all_inputs[i:i + 1]
            gray_inputs = all_gray_inputs[i:i+1]
            print('each rgb inputs shape ', inputs.shape)
            print('each gray inputs shape ', gray_inputs.shape)
            if len(gray_inputs.shape) == 4:
                gray_inputs = gray_inputs[0]
                inputs = inputs[0]

            # gray_inputs_t = transformation(gray_inputs)
            # gray_inputs_t = basic_transformation(gray_inputs, args["transform"])

            # gray_inputs_t1 = all_gray_inputs_tfomr1[i:i+1]
            # gray_inputs_t2 = all_gray_inputs_tfomr2[i:i+1]
            #gray_inputs_t1, gray_inputs_t2, gray_inputs_t3, gray_inputs_t4, gray_inputs_t5, gray_inputs_t6, gray_inputs_t7, gray_inputs_t8, gray_inputs_t9 = generate_transformation(gray_inputs)
            #gray_inputs_t1, gray_inputs_t2, gray_inputs_t3, gray_inputs_t4, gray_inputs_t5, gray_inputs_t6 = generate_transformation(gray_inputs)
            inputs_t1, inputs_t2, inputs_t3 = generate_transformation(inputs, i)
            if len(inputs_t1) == 4:
                inputs_t1 = inputs_t1[0]
                inputs_t2 = inputs_t2[0]
                inputs_t3 = inputs_t3[0]
        
            img_no += 1
            # print("gray inputs t ", gray_inputs_t)
            inputs_img = gen_image(inputs)
            inputs_t1_img = gen_image(inputs_t1)
            inputs_t2_img = gen_image(inputs_t2)
            inputs_t3_img = gen_image(inputs_t3)
        
            # gray_inputs_t4_img = gen_image(gray_inputs_t4)
            # gray_inputs_t5_img = gen_image(gray_inputs_t5)
            # gray_inputs_t6_img = gen_image(gray_inputs_t6)
            # gray_inputs_t7_img = gen_image(gray_inputs_t7)
            # gray_inputs_t8_img = gen_image(gray_inputs_t8)
            # gray_inputs_t9_img = gen_image(gray_inputs_t9)
            # print("saving........")
            # gray_inputs_t_img.save("t_img_{}.png".format(i))

            if args["htype"] == "phash":
                inputs_hash = imagehash.phash(inputs_img)
                inputs_t1_hash= imagehash.phash(inputs_t1_img)
                inputs_t2_hash= imagehash.phash(inputs_t2_img)
                inputs_t3_hash= imagehash.phash(inputs_t3_img)
                hash_differences_t1 = inputs_hash -  inputs_t1_hash
                hash_differences_t2 = inputs_hash -  inputs_t2_hash
                hash_differences_t3 = inputs_hash -  inputs_t3_hash
                # gray_inputs_t4_hash= imagehash.phash(gray_inputs_t4_img)
                # gray_inputs_t5_hash= imagehash.phash(gray_inputs_t5_img)
                # gray_inputs_t6_hash= imagehash.phash(gray_inputs_t6_img)
                # gray_inputs_t7_hash= imagehash.phash(gray_inputs_t7_img)
                # gray_inputs_t8_hash= imagehash.phash(gray_inputs_t8_img)
                # gray_inputs_t9_hash= imagehash.phash(gray_inputs_t9_img)
            elif args["htype"] == "blockhash":
                # if gray_inputs_img.mode == '1' or gray_inputs_img.mode == 'L' or gray_inputs_img.mode == 'P':
                #     im_original = gray_inputs_img.convert('RGB')
                #     im_t1 = gray_inputs_t1_img.convert('RGB')
                #     im_t2 = gray_inputs_t2_img.convert('RGB')
                #     im_t3 = gray_inputs_t3_img.convert('RGB')
                
                hash_differences_t1 = sum(1 for i, j in zip(robusthash.blockhash(inputs_img), robusthash.blockhash(inputs_t1_img)) if i != j)
                hash_differences_t2 = sum(1 for i, j in zip(robusthash.blockhash(inputs_img), robusthash.blockhash(inputs_t2_img)) if i != j)
                hash_differences_t3 = sum(1 for i, j in zip(robusthash.blockhash(inputs_img), robusthash.blockhash(inputs_t3_img)) if i != j)
            
            print('t1 hash differences for {}'.format(args["htype"]), hash_differences_t1)
            print('t2 hash differences for {}'.format(args["htype"]), hash_differences_t2)
            print('t3 hash differences for {}'.format(args["htype"]), hash_differences_t3)
            total_transformed_hash1.append(float(hash_differences_t1 / 64))
            total_transformed_hash2.append(float(hash_differences_t2 / 64))
            total_transformed_hash3.append(float(hash_differences_t3 / 64))
            total_transforms.append(float(hash_differences_t1 / 64))
            total_transforms.append(float(hash_differences_t2 / 64))
            total_transforms.append(float(hash_differences_t3 / 64))
        
            # gray_inputs_t2_hash= imagehash.phash(gray_inputs_t2_img)
            # hash_differences_t2 = gray_inputs_hash -  gray_inputs_t2_hash
            # print('transformation2 hash differences', (gray_inputs_hash -  gray_inputs_t2_hash))
            # print('transformation3 hash differences', (gray_inputs_hash -  gray_inputs_t3_hash))
            # print('transformation4 hash differences', (gray_inputs_hash -  gray_inputs_t4_hash))
            # print('transformation5 hash differences', (gray_inputs_hash -  gray_inputs_t5_hash))
            # print('transformation6 hash differences', (gray_inputs_hash -  gray_inputs_t6_hash))
            # print('transformation7 hash differences', (gray_inputs_hash -  gray_inputs_t7_hash))
            # print('transformation8 hash differences', (gray_inputs_hash -  gray_inputs_t8_hash))
            # print('transformation9 hash differences', (gray_inputs_hash -  gray_inputs_t9_hash))
        # total_transformed_hash1 = np.array(total_transformed_hash1)
        # total_transformed_hash2 = np.array(total_transformed_hash2)
        # total_transformed_hash3 = np.array(total_transformed_hash3)
        # total_transforms = np.array(total_transforms)
        # print("Just a quick test")
        # print("total ", img_no)
        # print("rotation ", np.mean(total_transformed_hash1), np.std(total_transformed_hash1)) 
        # print("cropped ", np.mean(total_transformed_hash2), np.std(total_transformed_hash2))  
        # print("aot ",np.mean(total_transformed_hash3), np.std(total_transformed_hash3))
        # print("total ", np.mean(total_transforms), np.std(total_transforms) )
        # print("minimum ", np.amin(total_transforms))
        # print("max ", np.amax(total_transforms))
        # print(total_transforms)

            if len(inputs.shape) == 3:
                inputs = inputs.reshape((1,)+inputs.shape)
                
            if len(inputs_t1.shape) == 3: 
                inputs_t1 = inputs_t1.reshape((1,)+inputs_t1.shape)
                inputs_t2 = inputs_t2.reshape((1,)+inputs_t2.shape)
                inputs_t3 = inputs_t3.reshape((1,)+inputs_t3.shape)
                # gray_inputs_t4 = gray_inputs_t4.reshape((1,)+gray_inputs_t4.shape)
                # gray_inputs_t5 = gray_inputs_t5.reshape((1,)+gray_inputs_t5.shape)
                # gray_inputs_t6 = gray_inputs_t6.reshape((1,)+gray_inputs_t6.shape)
                # gray_inputs_t7 = gray_inputs_t7.reshape((1,)+gray_inputs_t7.shape)
                # gray_inputs_t8 = gray_inputs_t8.reshape((1,)+gray_inputs_t8.shape)
                # gray_inputs_t9 = gray_inputs_t9.reshape((1,)+gray_inputs_t9.shape)

            best_modifier_path = './{}/best_modifier_img{}.npy'.format(args['save_ckpts'], i)
            timestart = time.time() 
            # only one picture at once
            #adv, const, L3 = attack.attack_batch(gray_inputs, inputs, i)
            #adv, const, L3, adv_current = attack.attack_batch(gray_inputs, inputs, gray_inputs_t1, gray_inputs_t2, gray_inputs_t3, gray_inputs_t4, gray_inputs_t5, gray_inputs_t6, gray_inputs_t7, gray_inputs_t8, gray_inputs_t9,i)
            adv, const, L3, adv_current, first_iteration= attack.attack_batch(gray_inputs, inputs, inputs_t1, inputs_t2, inputs_t3, i, best_modifier_path)
            timeend = time.time()
            if type(const) is list:
                const = const[0]
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            if len(adv_current.shape) == 3:
                adv_current = adv_current.reshape((1,)+ adv_current.shape)
    
            l2_distortion_direct = np.sum((adv - inputs) ** 2) ** 0.5
            print('l2_distortion between inputs and adv ', l2_distortion_direct)
            l2_distortion_current = np.sum((adv_current - inputs) ** 2) ** 0.5
            print('l2_distortion between inputs and adv_current ', l2_distortion_current)
            if len(inputs.shape) == 4:
                a,b,c = inputs[0].shape
                print("normalized a,b,c", a, b, c)
                l2_distortion_normalized = l2_distortion_direct / (a*b*c)**0.5
                l2_distortion_current_normalized = l2_distortion_current / (a*b*c)**0.5

            image0_ph = tf.placeholder(tf.float32)
            image1_ph = tf.placeholder(tf.float32)
            distance_t = lpips(image0_ph, image1_ph, model='net-lin', net='alex')
           
            # stacked_gray_inputs =  np.asarray(np.dstack((stacked_gray_inputs, stacked_gray_inputs, stacked_gray_inputs)))
            # stacked_adv =  np.asarray(np.dstack((stacked_adv, stacked_adv, stacked_adv)))
            # stacked_adv_current =  np.asarray(np.dstack((stacked_adv_current, stacked_adv_current, stacked_adv_current)))
            # if len(stacked_gray_inputs.shape) == 3:
            #     stacked_gray_inputs = stacked_gray_inputs.reshape((1,) + stacked_gray_inputs.shape)
            #     stacked_adv = stacked_adv.reshape((1,) + stacked_adv.shape)
            #     stacked_adv_current = stacked_adv_current.reshape((1,) + stacked_adv_current.shape)
        
            # print(stacked_adv.shape)
            # print(stacked_adv_current.shape)
            # perceptual distances
            with tf.Session(config=config) as session:
                distance1 = session.run(distance_t, feed_dict={image0_ph: inputs+0.5, image1_ph: adv+0.5})
                distance2 = session.run(distance_t, feed_dict={image0_ph: inputs+0.5, image1_ph: adv_current+0.5})

            distance1_normalized = distance1[0] / (a*b*c)**0.5
            distance2_normalized = distance2[0] / (a*b*c)**0.5
            success = False
            print("perceptual metrics distance between adv and img", distance1[0])
            print("perceptual metrics distance between adv_current and img", distance2[0])
            print("normalized perceptual metrics distance between adv and img", distance1_normalized)
            print("normalized perceptual metrics distance between adv_current and img", distance2_normalized)
            inputs_img = gen_image(inputs)
            adv_img = gen_image(adv)
           
            if args["htype"] == "phash":
                hash_differences = imagehash.phash(inputs_img, args['bits'], args['factor']) - imagehash.phash(adv_img, args['bits'], args['factor'])
                print('perceptual hash difference is ', hash_differences)
            elif args["htype"] == "blockhash":
                # if inputs_img.mode == '1' or inputs_img.mode == 'L' or inputs_img.mode == 'P':
                #     im_original = inputs_img.convert('RGB')
                #     im_adver = adv_img.convert('RGB')
                hash_differences = sum(1 for i, j in zip(robusthash.blockhash(inputs_img), robusthash.blockhash(adv_img)) if i != j)
                print('robust hash differences', hash_differences)
            
            

            # to verify hash differences with others
            # print('transformation1 hash differences with adv', (hash_differences))
            # print('transformation2 hash differences with adv', (adv_hash -  gray_inputs_t2_hash))
            # print('transformation3 hash differences with adv', (adv_hash -  gray_inputs_t3_hash))
            # print('transformation4 hash differences with adv', (adv_hash -  gray_inputs_t4_hash))
            # print('transformation5 hash differences with adv', (adv_hash -  gray_inputs_t5_hash))
            # print('transformation6 hash differences with adv', (adv_hash -  gray_inputs_t6_hash))
            # print('transformation7 hash differences with adv', (adv_hash -  gray_inputs_t7_hash))
            # print('transformation8 hash differences with adv', (adv_hash -  gray_inputs_t8_hash))
            # print('transformation9 hash differences with adv', (adv_hash -  gray_inputs_t9_hash))

            
            # if len(inputs.shape) == 4:
            #     a,b,c = inputs[0].shape
            #     l2_distortion_normalized = l2_distortion_direct / (a*b*c)**0.5
            #     l2_distortion_current_normalized = l2_distortion_current / (a*b*c)**0.5
            
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
                    suffix = "id{}_differ{}_{}_l2{:.2f}_pdistance{}".format(all_true_ids[i], hash_differences, success, l2_distortion_direct, distance1[0])
                    print("Saving to", suffix)
    
                    show(inputs, "{}/{}/{}_original_{}.png".format(args['save'], args['dataset'], img_no,suffix))
                    show(adv, "{}/{}/{}_adversarial_{}.png".format(args['save'], args['dataset'], img_no, suffix))
                    # show(adv - inputs, "{}/{}/{}_diff_{}.png".format(args['save'], args['dataset'], img_no, suffix))
            
                    print("[STATS][L1] total = {}, id = {}, time = {:.3f}, success = {}, const = {:.6f}, hash_avg={:.5f}, distortion = {:.5f}, success_rate = {:.3f}, l2_avg={:.5f}, p_avg={}, iteration_avg={}"
                    .format(img_no, all_true_ids[i], timeend - timestart, success, const, 0 if total_success == 0 else hash_total / total_success, l2_distortion_direct, total_success / float(img_no), 0 if total_success == 0 else l2_total / total_success, 0 if total_success == 0 else pdistance1_total/ total_success
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
                    print('robust hash differences', hash_differences)
                total_iterations += first_iteration
                hash_total2 += hash_differences_current
                l2_total2 += l2_distortion_current
                pdistance_total +=  distance2[0]
                print("Failed attacks!")
                suffix_current = "l2_{:.2f}_pdist{}_diff{}_success={}_time{}".format(l2_distortion_current, distance2[0], hash_differences_current, success, timeend - timestart)
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
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet", "maladv", "face", "random"], default="mnist")
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
                args['maxiter'] = 3000
            else:
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
    args['load_ckpt'] = './{}/best_modifier_img0.npy'.format(args['save_ckpts'])
    # when init_const is not specified, use a reasonable default
    # if args['init_const'] == 0.0:
    #     if args['binary_search']:
    #         args['init_const'] = 0.01
    #     else:
    #         args['init_const'] = 0.5
    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    print(args)
    main(args)
