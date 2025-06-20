import argparse
import os

import numpy as np
import torch
import torchvision
from diffusion import get_time_schedule, Posterior_Coefficients, \
    sample_from_model

from DWT_IDWT.DWT_IDWT_layer import IDWT_2D
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_wavelets import DWTInverse
from score_sde.models.ncsnpp_generator_adagn import NCSNpp, WaveletNCSNpp
import time

#wrdfkvni
# %%



def sample_and_test(args):
    torch.manual_seed(args.seed)
    device = 'cuda:0'

    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celebahq_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir

    def to_range_0_1(x):
        return (x + 1.) / 2.

    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    print(args.image_size, args.ch_mult, args.attn_resolutions)

    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))

    netG = gen_net(args).to(device)
    #ckpt = torch.load('./saved_info/wdd_gan/{}/{}/netG_{}.pth'.format(
    #    args.dataset, args.exp, args.epoch_id), map_location=device)
    #making sure the ceckpoints are loaded from folder 'checkpoints'

    #ckpt_path = "/content/drive/My Drive/Wavelettproject/checkpoints/netG_{}.pth".format(args.epoch_id)
    
    #ckpt_path = os.path.join("checkpoints", f"netG_{args.epoch_id}.pth")
    # to search for checkpoint not in the folder
    ckpt_path = f"netG_{args.epoch_id}.pth"
    ckpt = torch.load(ckpt_path, map_location=device)

    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)

    netG.load_state_dict(ckpt, strict=False)
    netG.eval()

    if not args.use_pytorch_wavelet:
        iwt = IDWT_2D("haar")
    else:
        iwt = DWTInverse(mode='zero', wave='haar').cuda()
    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    iters_needed = 40000 // args.batch_size                             # 50000 needs to be here
    print(f"Batch size: {args.batch_size}")
    print(f"Total iterations needed: {iters_needed}")

    #save_dir = "./wddgan_generated_samples/{}".format(args.dataset)
    # modifications to make sure it saves the folder and results to google drive
    #save_dir = "/content/drive/My Drive/Wavelettproject/wddgan_generated_samples/{}".format(args.dataset)
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    # new # modifications to make sure it is more robust
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "wddgan_generated_samples", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
  

    if args.measure_time:
        x_t_1 = torch.randn(args.batch_size, args.num_channels,
                            args.image_size, args.image_size).to(device)
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                fake_sample = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

                fake_sample *= 2.
                #didnt work for celeb_256
                #fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                    #(fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                try:
                  fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                    (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                except TypeError as e:
                    #print("⚠️ Failed using pytorch_wavelets-style IWT. Trying IDWT_2D-style call instead.")
                    fake_sample = iwt(
                      fake_sample[:, :3],   # LL
                      fake_sample[:, 3:6],  # LH
                      fake_sample[:, 6:9],  # HL
                      fake_sample[:, 9:12]  # HH
                      )

                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        print('🥹 Done with measuring inference time for all generated samples')
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        ng_output_path = os.path.join(save_dir, "timing_results.txt")

        with open(ng_output_path, "w") as f:
            f.write("GPU Inference time: {:.2f} +/- {:.2f} ms\n".format(mean_syn, std_syn))
        
        print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
        exit(0) #I want to know both FID and time simultaniously

    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():

                #if i == 0: #added block to see how much time it took to generate 1st batch
                    #start_time = time.time()

                
                x_t_1 = torch.randn(
                    args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)
                fake_sample = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

                fake_sample *= 2
                if not args.use_pytorch_wavelet:
                    fake_sample = iwt(
                        fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12])
                else: #didnt work for celeb_256
                    #fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                    #    (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                    try:
                      fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                        (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                    except TypeError as e:
                      #print("⚠️ Failed using pytorch_wavelets-style IWT. Trying IDWT_2D-style call instead.")
                      fake_sample = iwt(
                        fake_sample[:, :3],   # LL
                        fake_sample[:, 3:6],  # LH
                        fake_sample[:, 6:9],  # HL
                        fake_sample[:, 9:12]  # HH
                        )

                fake_sample = torch.clamp(fake_sample, -1, 1)

                fake_sample = to_range_0_1(fake_sample)  # 0-1
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j
                    torchvision.utils.save_image(
                        x, '{}/{}.jpg'.format(save_dir, index))
            if (i + 1) % 100 == 0:
                print(f"✅ {i + 1} batches generated")
                #if i==0:
                    #elapsed_time = time.time() - start_time
                    #print(f"⏱️ Time to generate 1st batch: {elapsed_time:.2f} seconds")
            

                        
                    
                #print('generating batch ', i)
        print('🫩Done generating batches')
        paths = [save_dir, real_img_dir]
        print(paths)

        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        print('😎Starting to calculate fid')
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('🥹Finished calculating fid')
        #print('FID = {}'.format(fid))
        #adding where it would write down fid results in a file
        fid_output_path = os.path.join(save_dir, "fid_results.txt")
        with open(fid_output_path, "w") as f:
            f.write("FID = {:.4f}\n".format(fid))
        print('FID = {:.4f} (saved to {})'.format(fid, fid_output_path))

        grid_output_path = os.path.join(save_dir, "samples_{}.jpg".format(args.dataset))
        torchvision.utils.save_image(fake_sample, grid_output_path, nrow=8, padding=0)
        print(f"Image grid is haved to: {grid_output_path}")
        #



    #else: #I also want after computing FID to get that image concatination
        x_t_1 = torch.randn(args.batch_size, args.num_channels,
                            args.image_size, args.image_size).to(device)
        fake_sample = sample_from_model(
            pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

        fake_sample *= 2
        if not args.use_pytorch_wavelet:
            fake_sample = iwt(
                fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12])
        else:
            fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
        fake_sample = torch.clamp(fake_sample, -1, 1)

        fake_sample = to_range_0_1(fake_sample)  # 0-1
        torchvision.utils.save_image(
            fake_sample, './samples_{}.jpg'.format(args.dataset), nrow=8, padding=0)
        print("Image grid is also in this file: samples_{}.jpg".format(args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--measure_time', action='store_true', default=False,
                        help='whether or not measure time')
    parser.add_argument('--epoch_id', type=int, default=1000)
    parser.add_argument('--num_channels', type=int, default=12,
                        help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), type=int, nargs='+',
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # generator and training
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy',
                        help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=100,# default was 200, i made it 64 as RAM was exxiding
                        help='sample generating batch size')

    # wavelet GAN
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--net_type", default="normal")
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    args = parser.parse_args()
    #these two make sure that fid and inference time are always computed  
    args.measure_time = False
    #args.compute_fid = True
    
    # time=Ture fid=False gives only time
    # time=false fid=true gives fid and grid
    sample_and_test(args)

