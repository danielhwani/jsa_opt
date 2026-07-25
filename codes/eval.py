#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import os
from config import config
import utils.utils_image as util_image
import utils.utils_rend_img as util_rend
from pytz import timezone
import datetime
import preprocess as pre
import numpy as np
import loss as L
import pyexr
import time
import math
import torch.nn.functional as F


def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask

def _cosine_ramp(length):
    # 0 -> 1 raised-cosine (Hann half-window); two of these mirrored against each
    # other across an overlap region sum to exactly 1 at every pixel, so blending
    # neighboring tiles with this window reconstructs a seamless, non-uniform average.
    if length <= 0:
        return torch.zeros(0)
    if length == 1:
        return torch.ones(1)
    return 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, length))


def _axis_weight(size, overlap, is_first, is_last):
    # full weight (1.0) on edges that touch the true image border (no neighbor to
    # blend with there); cosine taper only on edges shared with a neighboring tile.
    w = torch.ones(size)
    ramp_len = min(overlap, size // 2)
    if ramp_len > 0:
        ramp = _cosine_ramp(ramp_len)
        if not is_first:
            w[:ramp_len] = ramp
        if not is_last:
            w[-ramp_len:] = ramp.flip(0)
    return w


def _tile_starts(size, tile_size, stride):
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def tiled_forward(net, x, y, tile_size, device, overlap=0):
    # Each tile is forwarded through the full U-Net independently (4x downsample +
    # win_size=8 window attention hardcoded in main_train.py), so tile_size itself
    # must be a multiple of 128 or window_partition() crashes inside individual tiles.
    assert tile_size % 128 == 0, (
        "eval_tile_size (%d) must be a multiple of 128: the model downsamples 4x "
        "with win_size=8 window attention, so each independently-forwarded tile "
        "needs a spatial size that is a multiple of 128." % tile_size
    )
    assert 0 <= overlap < tile_size, "eval_tile_overlap must be in [0, tile_size)"

    B, _, H, W = x.shape

    # tiles must fully fit inside the image at least once
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        y = F.pad(y, (0, pad_w, 0, pad_h), 'reflect')

    Hp, Wp = x.shape[2], x.shape[3]
    stride = tile_size - overlap
    h_starts = _tile_starts(Hp, tile_size, stride)
    w_starts = _tile_starts(Wp, tile_size, stride)

    out_sum = None
    weight_sum = torch.zeros(1, 1, Hp, Wp)

    for i in h_starts:
        for j in w_starts:
            tile_x = x[:, :, i:i + tile_size, j:j + tile_size]
            tile_y = y[:, :, i:i + tile_size, j:j + tile_size]

            with torch.no_grad():
                tile_out = net(tile_x, tile_y)

            # move off GPU immediately so VRAM doesn't accumulate across tiles
            tile_out = tile_out.detach().to("cpu")
            if out_sum is None:
                out_sum = torch.zeros(B, tile_out.shape[1], Hp, Wp, dtype=tile_out.dtype)

            wh = _axis_weight(tile_size, overlap, i == 0, i == Hp - tile_size)
            ww = _axis_weight(tile_size, overlap, j == 0, j == Wp - tile_size)
            weight2d = (wh.unsqueeze(1) * ww.unsqueeze(0)).to(tile_out.dtype)  # [tile_size, tile_size]

            out_sum[:, :, i:i + tile_size, j:j + tile_size] += tile_out * weight2d
            weight_sum[:, :, i:i + tile_size, j:j + tile_size] += weight2d

            del tile_out, tile_x, tile_y
            torch.cuda.empty_cache()

    out = out_sum / weight_sum.clamp(min=1e-8)
    out = out[:, :, :H, :W].to(device)
    return out

def eval_train(net, test_loader, epoch):
    # evaluate network
    net.eval()
    test_cnt = 0
    Loss_average = 0.0
    PSNR_average = 0.0     
    RMSE_average = 0.0     
    nan_count = 0.0

    # setting for eval
    test_dir = os.path.join(config["data_dir"],config["task"])
    util_image.mkdir(test_dir)
    save_dir = os.path.join(test_dir,"output")
    util_image.mkdir(save_dir)

    # make log
    log_name = config["task"]
    logfile_ours = open(os.path.join(test_dir,'loss_ours_%s_%d.txt') % (log_name ,epoch), 'a+')
    
    if config["multi_gpu"] == False:
        device = torch.device("cuda:0")
        net = net.to(device)

    for iteration, (input, gt, path) in enumerate(test_loader): 
        # img name
        str_path = str(path)
        image_name_ext = os.path.basename(str_path)
        img_name, ext = os.path.splitext(image_name_ext)

        # preprocessing
        aux_features = input['aux']
        aux_features[:, :, :, 3:6] = torch.FloatTensor(util_rend.preprocess_normal(aux_features[:, :, :, 3:6]))  
        aux_features = aux_features.permute(0, 3, 1, 2)

        color_noisy = input["color"]
        color_noisy = util_rend.preprocess_specular(color_noisy)
        color_noisy = color_noisy.permute(0, 3, 1, 2)

        color_gt = gt['color']

        color_gt_for_loss = util_rend.preprocess_specular(color_gt)
        color_gt_for_loss = color_gt_for_loss.permute(0, 3, 1, 2)
        if config["multi_gpu"]:
            color_gt_for_loss = color_gt_for_loss.to("cuda")
        else:
            color_gt_for_loss = color_gt_for_loss.to(device)          
        color_gt = color_gt.permute(0, 3, 1, 2)
        
        input_tensor = torch.cat([color_noisy, aux_features], dim = 1)

        # padding for u-net sturcture
        factor = 128
        h,w = input_tensor.shape[2], input_tensor.shape[3]
        hh,ww = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        xx = max(hh,ww)
        padh = xx-h 
        padw = xx-w 

        color_noisy =  F.pad(color_noisy, (0,padw,0,padh), 'reflect')
        aux_features =  F.pad(aux_features, (0,padw,0,padh), 'reflect')

        if config["multi_gpu"]:
            x = color_noisy
            y = aux_features
        else:
            x = color_noisy.to(device)
            y = aux_features.to(device)

        tile_size = config.get("eval_tile_size", config.get("patch_size", 128))
        tile_overlap = config.get("eval_tile_overlap", 0)
        out = tiled_forward(net, x, y, tile_size, device, overlap=tile_overlap)

        out = out[:,:,:h,:w]
        loss = torch.mean(L.RelL2(out, color_gt_for_loss))

        # inverse log transform
        output_c_n = util_rend.postprocess_specular(out.data.cpu().numpy()[0])
        gt_c_n = color_gt.numpy()[0]

        noisy_c_n_255  = util_rend.tensor2img(color_noisy.cpu().numpy()[0], post_spec=True)
        output_c_n_255 = util_rend.tensor2img(out.data.cpu().numpy()[0], post_spec=True)
        gt_c_n_255     = util_rend.tensor2img(color_gt.cpu().numpy()[0])

        # rmse: output after post-processing, without tone mapping and * 255 (use postprocess_specular/ postprocess_diffuse)
        # psnr, ssim: output after post-processing, tone mapping and * 255 (use tensor2img)
        # metrics are computed on the reconstructed full image (post-stitching), not per-tile
        our_rmse = util_image.calculate_rmse(np.transpose(output_c_n.copy(), (1, 2, 0)), np.transpose(gt_c_n.copy(), (1, 2, 0)))
        our_psnr = util_image.calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
        our_ssim = util_image.calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())

        if np.isnan(our_rmse):
            nan_count += 1
        else:
            RMSE_average += our_rmse

        # save output images
        filename = "%s_epoch%04d_%s%.4f.png" % (img_name, epoch, 'PSNR', our_psnr)
        img_save_path = os.path.join(save_dir, filename)
        util_image.imwrite(output_c_n_255, img_save_path)

        # updata info
        Loss_average += loss
        PSNR_average += our_psnr    
        test_cnt = test_cnt + 1

        # log eval info per image
        str_print_test_img = "%20s  -  rel l2 Loss: %0.6f  PSNR: %0.6f  SSIM: %0.6f  RMSE: %0.6f\n" %(img_name, loss.item(),our_psnr, our_ssim, our_rmse)
        logfile_ours.write(str_print_test_img)
        logfile_ours.flush()

    # cal avg 
    Loss_average /= test_cnt
    PSNR_average /= test_cnt

    # log eval info of average value
    if config['timezone'] == None:
        str_data_eval = datetime.datetime.now().astimezone(None).strftime("%Y-%m-%d %H:%M:%S")
    else:
        str_data_eval = datetime.datetime.now().astimezone(timezone(config["timezone"])).strftime("%Y-%m-%d %H:%M:%S")
        
    str_print_evalutation = "[Epoch %03d] (%s) rel l2 Loss avg: %0.6f   PSNR avg: %0.6f\n" %(epoch, str_data_eval, Loss_average, PSNR_average)
    print(str_print_evalutation)
    logfile_ours.write(str_print_evalutation)
    logfile_ours.flush()

    return Loss_average, PSNR_average

def eval_test(net, test_loader, epoch):
    # evaluate network
    net.eval()
    test_cnt = 0
    Loss_average = 0.0
    PSNR_average = 0.0     
    RMSE_average = 0.0     
    time_average = 0.0
    nan_count = 0.0

    # setting for eval
    test_dir = os.path.join(config["data_dir"],config["task"])
    util_image.mkdir(test_dir)
    save_dir = os.path.join(test_dir,"output_test")
    util_image.mkdir(save_dir)

    # make log
    log_name = config["task"]
    if type(epoch) is str: 
        logfile_ours = open(os.path.join(test_dir,'loss_ours_%s_%s.txt') % (log_name ,epoch), 'a+')
    elif type(epoch) is int: 
        logfile_ours = open(os.path.join(test_dir,'loss_ours_%s_%d.txt') % (log_name ,epoch), 'a+')

    if config["multi_gpu"] == False:
        device = torch.device("cuda:0")
        net = net.to(device)

    for iteration, (input, gt, path) in enumerate(test_loader): 
        # img name
        str_path = str(path)
        image_name_ext = os.path.basename(str_path)
        img_name, ext = os.path.splitext(image_name_ext)

        # preprocessing
        aux_features = input['aux']
        aux_features[:, :, :, 3:6] = torch.FloatTensor(util_rend.preprocess_normal(aux_features[:, :, :, 3:6]))  
        aux_features = aux_features.permute(0, 3, 1, 2)

        color_noisy = input["color"]
        color_noisy = util_rend.preprocess_specular(color_noisy)
        color_noisy = color_noisy.permute(0, 3, 1, 2)

        color_gt = gt['color']

        color_gt_for_loss = util_rend.preprocess_specular(color_gt)
        color_gt_for_loss = color_gt_for_loss.permute(0, 3, 1, 2)
        if config["multi_gpu"]:
            color_gt_for_loss = color_gt_for_loss.to("cuda")
        else:
            color_gt_for_loss = color_gt_for_loss.to(device)  
        
        color_gt = color_gt.permute(0, 3, 1, 2)
        
        input_tensor = torch.cat([color_noisy, aux_features], dim = 1)

        # padding for u-net sturcture
        factor = 128
        h,w = input_tensor.shape[2], input_tensor.shape[3]
        hh,ww = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        xx = max(hh,ww)
        padh = xx-h 
        padw = xx-w 
        
        color_noisy =  F.pad(color_noisy, (0,padw,0,padh), 'reflect')
        aux_features =  F.pad(aux_features, (0,padw,0,padh), 'reflect')

        if config["multi_gpu"]:
            x = color_noisy
            y = aux_features
        else:
            x = color_noisy.to(device)
            y = aux_features.to(device)

        # old
        # torch.cuda.synchronize()
        # start_time = time.time()
        # with torch.no_grad():
        #     out = net(x, y)
        # torch.cuda.synchronize()
        # end_time = time.time() - start_time

        # new
        tile_size = config.get("eval_tile_size", config.get("patch_size", 128))
        tile_overlap = config.get("eval_tile_overlap", 0)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)

        start_event.record()
        out = tiled_forward(net, x, y, tile_size, device, overlap=tile_overlap)
        end_event.record()

        end_event.synchronize()
        end_time = start_event.elapsed_time(end_event) * 1e-3  # seconds

        time_average += end_time
        out = out[:,:,:h,:w]
        loss = torch.mean(L.RelL2(out, color_gt_for_loss))

        # inverse log transform
        output_c_n = util_rend.postprocess_specular(out.data.cpu().numpy()[0])        
        gt_c_n = color_gt.numpy()[0]

        noisy_c_n_255  = util_rend.tensor2img(color_noisy.cpu().numpy()[0], post_spec=True)
        output_c_n_255 = util_rend.tensor2img(out.data.cpu().numpy()[0], post_spec=True)
        gt_c_n_255     = util_rend.tensor2img(color_gt.cpu().numpy()[0])

        # rmse: output after post-processing, without tone mapping and * 255 (use postprocess_specular/ postprocess_diffuse)
        # psnr, ssim: output after post-processing, tone mapping and * 255 (use tensor2img)
        our_rmse = util_image.calculate_rmse(np.transpose(output_c_n.copy(), (1, 2, 0)), np.transpose(gt_c_n.copy(), (1, 2, 0)))
        our_psnr = util_image.calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
        our_ssim = util_image.calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())
        if np.isnan(our_rmse):
            nan_count += 1
        else:
            RMSE_average += our_rmse

        # save output images
        if type(epoch) is str: 
            filename = "%s_epoch%s_%s%.4f.png" % (img_name, epoch, 'PSNR', our_psnr)
            filename_gt = "%s_GT_epoch%s_%s%.4f.png" % (img_name, epoch, 'PSNR', our_psnr)
        elif type(epoch) is int: 
            filename = "%s_epoch%04d_%s%.4f.png" % (img_name, epoch, 'PSNR', our_psnr)
            filename_gt = "%s_GT_epoch%04d_%s%.4f.png" % (img_name, epoch, 'PSNR', our_psnr)
        
        img_save_path = os.path.join(save_dir, filename)
        util_image.imwrite(output_c_n_255, img_save_path)
        
        img_save_path_gt = os.path.join(save_dir, filename_gt)
        util_image.imwrite(gt_c_n_255, img_save_path_gt)

        output_c_n_exr = np.transpose(output_c_n, (1, 2, 0))
        if type(epoch) is str: 
            filename = "%s_epoch%s_%s%.4f.exr" % (img_name, epoch, 'PSNR', our_psnr)
            filename_gt = "%s_GT_epoch%s_%s%.4f.exr" % (img_name, epoch, 'PSNR', our_psnr)
        elif type(epoch) is int: 
            filename = "%s_epoch%04d_%s%.4f.exr" % (img_name, epoch, 'PSNR', our_psnr)
            filename_gt = "%s_GT_epoch%04d_%s%.4f.exr" % (img_name, epoch, 'PSNR', our_psnr)
        
        img_save_path = os.path.join(save_dir, filename)
        pyexr.write(img_save_path, output_c_n_exr)

        gt_c_n_exr = np.transpose(gt_c_n, (1, 2, 0))
        img_save_path_gt = os.path.join(save_dir, filename_gt)    
        pyexr.write(img_save_path_gt, gt_c_n_exr)

        # updata info
        Loss_average += loss
        PSNR_average += our_psnr    
        test_cnt = test_cnt + 1

        # log eval info per image
        str_print_test_img = "%20s  -  rel l2 loss: %0.6f  PSNR: %0.6f  SSIM: %0.6f  RMSE: %0.6f (%f sec)\n" %(img_name, loss.item(),our_psnr, our_ssim, our_rmse, end_time)
        print(str_print_test_img)
        logfile_ours.write(str_print_test_img)
        logfile_ours.flush()

    # cal avg 
    Loss_average /= test_cnt
    PSNR_average /= test_cnt
    time_average /= test_cnt

    # log eval info of average value
    if config['timezone'] == None:
        str_data_eval = datetime.datetime.now().astimezone(None).strftime("%Y-%m-%d %H:%M:%S")
    else:
        str_data_eval = datetime.datetime.now().astimezone(timezone(config["timezone"])).strftime("%Y-%m-%d %H:%M:%S")

    if type(epoch) is str: 
        str_print_evalutation = "[Epoch %s] (%s) rel l2 loss avg: %0.6f   PSNR avg: %0.6f Time avg: %0.6f\n" %(epoch, str_data_eval, Loss_average, PSNR_average, time_average)
    elif type(epoch) is int:
        str_print_evalutation = "[Epoch %03d] (%s) rel l2 loss avg: %0.6f   PSNR avg: %0.6f Time avg: %0.6f\n" %(epoch, str_data_eval, Loss_average, PSNR_average, time_average)

    print(str_print_evalutation)
    logfile_ours.write(str_print_evalutation)
    logfile_ours.flush()

    return Loss_average, PSNR_average