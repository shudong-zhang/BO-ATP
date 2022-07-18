import argparse
import os
import pickle

import numpy as np
import torch
# torch.cuda.current_device()

from bayesopt import Bayes_opt
from utilities.upsampler import upsample_projection
from utilities.utilities import get_init_data
from dataset import load_mnist_test_data, load_cifar10_test_data, load_imagenet_test_data

from classifers import mnist_model,vgg,resnet,cifar_resnet
from general_torch_model import GeneralTorchModel
from loss_function import *
import torch.nn.functional as F
import torchvision.models as models
from FCN import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
parser.add_argument('-f', '--func', help='Objective function(datasets): mnist, cifar10, imagenet',
                    default='imagenet', type=str)
parser.add_argument('-m', '--model', help='Surrogate model: GP or ADDGPLD or ADDGPFD or GPLDR',
                    default='GP', type=str)
parser.add_argument('-mn', '--model_name', default='Resnet50',choices=['mnist_t1','mnist_t2','cifar_vgg19',
                                                                'cifar_resnet','cifar_at','cifar_trades',
                                                                'Resnet50','VGG19'])
parser.add_argument('-acq', '--acq_func', help='Acquisition function type: LCB, EI',
                    default='LCB', type=str)
parser.add_argument('-bm', '--batch_opt', help='BO batch option: CL, KB',
                    default='CL', type=str)
parser.add_argument('-b', '--batch_size', help='BO batch size.',
                    default=1, type=int)
parser.add_argument('-ld', '--low_dim', help='Dimension of reduced subspace.',
                    default=196, type=int)
parser.add_argument('-init', '--n_init', help='Initial number of observation.',
                    default=10, type=int)
parser.add_argument('-nitr', '--max_itr', help='Max BO iterations.',
                    default=990, type=int)
parser.add_argument('-rd', '--reduction', help='Use which dimension reduction technique. '
                                               'BILI, BICU, NN, CLUSTER, None.',
                    default='BILI', type=str)
parser.add_argument('-ni', '--nimgs', help='Number of images to be attacked '
                                           'Set to 1000 for MNIST,CIFAR10 and ImageNet',
                    default=1000, type=int)
parser.add_argument('-sp', '--sparse', help='Sparse GP method: subset selection (SUBRAND, SUBGREEDY), '
                                            'subset selection for decomposition/low-dim learning only (ADDSUBRAND), '
                                            'subset selection for both (SUBRANDADD)',
                    default='None', type=str)
parser.add_argument('-dis', '--dis_metric',
                    help='Distance metric for cost aware BO. None: normal BO, 2: exp(L2 norm),'
                         '10: exp(L_inf norm)',
                    default=None, type=int)
parser.add_argument('-ta', '--target_attack', help='Metric used to compute objective function.',
                    default='untarget', type=str,choices=['untarget','target'])
parser.add_argument('-tl','--target_label',type=int,default=0)
parser.add_argument('-freq', '--update_freq', help='Frequency to update the surrogate hyperparameters.',
                    default=5, type=int)
parser.add_argument('-nsub', '--nsubspace',
                    help='Number of subspaces to be decomposed into only applicable for ADDGP: '
                         'we set to 12 for CIFAR10 or MNIST and 27 for ImageNet',
                    default=12, type=int)
parser.add_argument('-se', '--seed', help='Random seed', default=1234, type=int)
parser.add_argument('--results', type=str, default='./attack_results.txt')
parser.add_argument('--epsilon',type=float,default=0.07)
parser.add_argument('-g','--greedy',type=str,default='greedy')
parser.add_argument('--bound',type=int,default=1)
parser.add_argument('--defense',type=str,default='na',choices=['jpeg','denoise','na'])

args = parser.parse_args()
print(f"Got arguments: \n{args}")
obj_func = args.func
model_type = args.model
model_name = args.model_name
acq_func = args.acq_func
batch_opt = args.batch_opt
batch_n = args.batch_size
n_itrs = args.max_itr
nimgs = args.nimgs
epsilon = args.epsilon
greedy = True if args.greedy == 'greedy' else False

dim_reduction = args.reduction
low_dim = args.low_dim

n_init = args.n_init
dis_metric = args.dis_metric
target = True if args.target_attack=='target' else False
target_label = np.array([args.target_label])
update_freq = args.update_freq
nsubspace = args.nsubspace
sparse = args.sparse
seed = args.seed
bound = args.bound
defense = args.defense

directory = './'
if obj_func == 'mnist':
    epsilon = 0.3
    high_dim = int(28 * 28)
    nlabels = 10
    nchannel = 1
    dataloader = load_mnist_test_data()
    if model_name == 'mnist_t1':
        model = mnist_model.MNIST_target_1().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('checkpoints/mnist_t1.pth')['net'])
        model.eval()
    elif model_name == 'mnist_t2':
        model = mnist_model.MNIST_target_2().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('checkpoints/mnist_t2.pth')['net'])
        model.eval()
    model = GeneralTorchModel(model,defense, n_class=nlabels, im_mean=None, im_std=None)
    encoder = MNIST_Encoder()
    decoder = MNIST_Decoder()
    weight = torch.load('vae/MNIST_target_9.pth') if target else torch.load('vae/MNIST_untarget.pth')
elif obj_func == 'cifar10':
    high_dim = int(32 * 32)
    nchannel = 3
    nlabels = 10
    dataloader = load_cifar10_test_data()
    if model_name == 'cifar_vgg19':
        model = vgg.VGG("VGG19").cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('checkpoints/cifar_vgg19_ckpt.pth')['net'])
        model = GeneralTorchModel(model, defense,n_class=nlabels, im_mean=[0.4914, 0.4822, 0.4465],im_std=[0.2023, 0.1994, 0.2010])
    elif model_name == 'cifar_resnet':
        model = cifar_resnet.ResNet18()
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load('checkpoints/cifar_resnet18_ckpt.pth')['net'])
        model = GeneralTorchModel(model, defense,n_class=nlabels, im_mean=[0.4914, 0.4822, 0.4465],im_std=[0.2023, 0.1994, 0.2010])
    encoder = CIFAR10_Encoder()
    decoder = CIFAR10_Decoder()
    # weight = torch.load('/data/zsd/pytorch-train-generator/ae_checkpoint/linf/best_success_CIFAR_vgg16_senet_densenet_target_0_linf_8x8.pth') if target else torch.load('/data/zsd/pytorch-train-generator/ae_checkpoint/linf/best_success_CIFAR_vgg16_senet_densenet_untarget_linf_8x8.pth')
    # weight = torch.load('/data/zsd/pytorch-train-generator/ae_checkpoint/linf/best_success_CIFAR_vgg16_senet_densenet_tanh_target_0.pth') if target else torch.load('/data/zsd/pytorch-train-generator/ae_checkpoint/linf/best_success_CIFAR_vgg16_senet_densenet_tanh_untarget_linf.pth')
    weight = torch.load('checkpoints/CIFAR_target_0_linf.pth') if target else torch.load('checkpoints/CIFAR_untarget_linf.pth')

elif obj_func == 'imagenet':
    high_dim = int(224 * 224)
    nchannel = 3
    nlabels = 1000
    dataloader = load_imagenet_test_data()
    if model_name == 'Resnet50':
        model = models.resnet50(pretrained=True).cuda()
    elif model_name == 'Resnet34':
        model = models.resnet34(pretrained=True).cuda()
    elif model_name == 'VGG19':
        model = models.vgg19_bn(pretrained=True).cuda()
    # elif model_name == 'jpeg':
    model = torch.nn.DataParallel(model)
    model = GeneralTorchModel(model, n_class=nlabels, im_mean=[0.485, 0.456, 0.406],
                              im_std=[0.229, 0.224, 0.225],defense_method='gauss')

    encoder = Imagenet_Encoder()
    decoder = Imagenet_Decoder()
    if target:
        weight = torch.load('checkpoints/imagenet_target_0_linf.pth')
        # weight = torch.load('/data/zsd/pytorch-train-generator/ae_checkpoint/linf/Imagenet_VGG16_Resnet18_Squeezenet_Googlenet_target_0_8x8.pytorch',map_location="cuda:0")
    else:
        weight = torch.load('checkpoints/imagenet_untarget_linf.pth')
# load autoencoder
encoder_weight = {}
decoder_weight = {}
for key, val in weight.items():
    if key.startswith('0.'):
        encoder_weight[key[2:]] = val
    elif key.startswith('1.'):
        decoder_weight[key[2:]] = val
encoder.load_state_dict(encoder_weight)
decoder.load_state_dict(decoder_weight)
encoder.cuda().eval()
decoder.cuda().eval()

# Specify the experiment results saving directory
results_data_folder = f'{directory}exp_results/{obj_func}_tf_{model_type}_ob{target}_' \
                      f'_freq{update_freq}_ld{low_dim}_{dim_reduction}/'
if not os.path.exists(results_data_folder):
    os.makedirs(results_data_folder)

# Define the BO objective function

function = Function(model, margin=5, nlabels=nlabels,target=target)

# def project(x,delta,epsilon):
#     x_adv = x + delta
#     x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
#     x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     return x_adv

def BayesOpt_attack(model, function, img_id, image, label, model_type, acq_type, batch_size, low_dim, sparse,
                    seed, n_init=50, num_iter=40, dim_reduction='BILI',encoder = None,decoder = None,
                    latent=None, cost_metric=None, update_freq=10, nsubspaces=1,x_bound = bound):


    ori_image = image.clone()
    if latent is None:
        latent = encoder(image).squeeze().view(-1)
    dim = len(latent)
    x_bounds = np.vstack([[-x_bound, x_bound]] * dim)
    latent = latent.unsqueeze(0)
    perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*epsilon, -epsilon,epsilon)
    pred_label = model.predict_label(torch.clamp(ori_image+perturbation,0,1))
    function.current_counts +=1
    if (target==False and pred_label != label) or (target == True and pred_label == label):
        return True
    
    # logits
    def f_logits(x,image_x,min_loss):
        x = torch.from_numpy(x).float().cuda()
        x_upsample = torch.clamp(decoder(latent + x).squeeze(0)*epsilon, -epsilon, epsilon)

        # delta = torch.clamp(decoder(latent + x).squeeze(0)*epsilon, -epsilon, epsilon)
        with torch.no_grad():
            loss = function(torch.clamp(image_x + x_upsample, 0, 1), label)
            # loss = function(torch.clamp(image_x + eta*delta, 0, 1), label)
        loss = loss[1].detach().cpu().numpy()
        if greedy:
            if min_loss is None:
                min_loss = loss
            else:
                if loss < min_loss:
                    min_loss = loss
                    image_x = image_x + x_upsample
                    image_x = torch.clamp(torch.min(torch.max(image_x,ori_image-epsilon),ori_image+epsilon),0,1)

        return loss,image_x,min_loss
    # Define the name of results file and failure fail(for debug or resume)
    results_file_name = os.path.join(results_data_folder,
                                     f'{model_type}{acq_type}{batch_size}_{dim_reduction}_'
                                     f'd{low_dim}_i{label}_id{img_id}')
    failed_file_name = os.path.join(results_data_folder,
                                    f'failed_{model_type}{acq_type}{batch_size}_{dim_reduction}_'
                                    f'd{low_dim}_i{label}_id{img_id}')

    X_opt_all_slices = []
    Y_opt_all_slices = []
    X_query_all_slices = []
    Y_query_all_slices = []
    X_reduced_opt_all_slices = []
    X_reduced_query_all_slices = []

    # Specify the random seed
    np.random.seed(seed)

    # Generate initial observation data for BO

    print('generate new init data')
    x_init = np.empty([n_init,x_bounds.shape[0]])
    y_init = np.empty([n_init,1])
    x_init_img = image
    min_loss = None
    for i in range(n_init):
        x_init_, y_init_,x_init_img,min_loss = get_init_data(obj_func=f_logits, n_init=1, bounds=x_bounds,function_args=[x_init_img,min_loss])		
        x_init[i] = x_init_
        y_init[i] = y_init_
        if min_loss < 0:
            return True
    print("x_init.shape=", x_init.shape)
    print("y_init.shape=", y_init.shape)

    # Initialise BO
    
    bayes_opt = Bayes_opt(func=f_logits, bounds=x_bounds, saving_path=failed_file_name,func_args=[x_init_img,min_loss])
    
    bayes_opt.initialise(X_init=x_init, Y_init=y_init, model_type=model_type, acq_type=acq_type,
                         sparse=sparse, nsubspaces=nsubspaces, batch_size=batch_size, update_freq=update_freq,
                         nchannel=nchannel, high_dim=high_dim, dim_reduction=dim_reduction,
                         cost_metric=cost_metric, seed=seed)

    # Run BO
    X_query_full, Y_query, X_opt_full, Y_opt, success, time_record = bayes_opt.run(total_iterations=num_iter)

    '''
    Visualize the output: ori image, adv_image

    ori_img = image.detach().cpu().numpy()
    ori_img = ori_img.squeeze()
    ori_img = np.transpose(ori_img, (1, 2, 0))
    plt.imshow(ori_img)
    plt.show()

    x_opt = X_opt_full[np.argmin(Y_opt)]
    x_opt_tensor = torch.from_numpy(x_opt).float().cuda()
    x_low = x_opt_tensor.view(1,nchannel,int(np.sqrt(low_dim)),int(np.sqrt(low_dim)))
    x_upsample = F.interpolate(x_low,size=(int(np.sqrt(high_dim)),int(np.sqrt(high_dim))),mode='bilinear')
    x_upsample = x_upsample*epsilon
    adv = torch.clamp(image + x_upsample,0,1)
    adv_np = adv.detach().cpu().numpy()


    adv_np= adv_np.squeeze()
    adv_np = np.transpose(adv_np, (1, 2, 0))
    plt.imshow(adv_np)
    plt.show()
    '''

    return success[0]

def read_results(image_resluts):
    iaa = []
    lines = open(image_resluts,'r').readlines()
    for img in lines:
        a = int(img.strip().split(" ")[0][6:])
        iaa.append(a)
    return iaa

count_success = 0
count_total = 0
count_average = []

if os.path.exists("{}_{}_target_{}_{}_greedy_{}_gauss.txt".format(obj_func,model_name,target,epsilon,greedy))== True:
    img_already_attack = read_results("{}_{}_target_{}_{}_greedy_{}_gauss.txt".format(obj_func,model_name,target,epsilon,greedy))
else:
    img_already_attack = []

for img_id, (image, label) in enumerate(dataloader):
    if img_id in img_already_attack:
        print("image {} already attacked, pass".format(img_id))
        continue
    image = image.cuda()
    label = label.cuda()
    if target:
        label = torch.from_numpy(target_label).cuda()
    predict_label = model.predict_label(image)
    # function.current_counts = 1
    if (predict_label == label and target==False) or (predict_label != label and target==True):
        torch.cuda.empty_cache()
        success = BayesOpt_attack(model=model, function=function, img_id=img_id, image=image, label=label,
                                    model_type=model_type,
                                    acq_type=acq_func, batch_size=batch_n, low_dim=low_dim, sparse=sparse, seed=seed,
                                    n_init=n_init,
                                    num_iter=n_itrs, dim_reduction=dim_reduction,encoder = encoder,decoder = decoder,
                                    latent=None, cost_metric=dis_metric,
                                    update_freq=update_freq, nsubspaces=nsubspace,x_bound=bound)

        # print(success)
        count_success += int(success)
        count_total += 1
        if success:
            count_average.append(function.current_counts)
        print("image: {} eval_count: {} success: {} average_count: {} success_rate: {} "
              .format(img_id, function.current_counts, success, np.mean(np.array(count_average)),
                      float(count_success) / float(count_total)))
        
        # if function.current_counts != 1:
        with open("{}_{}_target_{}_{}_greedy_{}_gauss.txt".format(obj_func,model_name,target,epsilon,greedy),'a') as f:
                f.write("image:{} success:{} query:{}".format(img_id,success,function.current_counts)+'\n')
        # f.close()
        # else:
        # 	count_total = count_total-1

        function.new_counter()
        if count_total == nimgs:
            break
success_rate = float(count_success) / float(count_total)
print("success rate {}".format(success_rate))
print("average eval count {}".format(np.mean(np.array(count_average))))

with open(args.results, 'a') as f:
    f.write("dataset: {} model: {} success rate: {} average eval count: {} '\n'".format(
        obj_func, model_name, success_rate, np.mean(np.array(count_average))
    ))
