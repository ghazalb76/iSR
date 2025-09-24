#!/usr/local/bin/python
import sys
import torch
import time
import numpy as np
import argparse
import model.SICNet as SICNet
import config
from torch.utils.data import DataLoader
from myUtils import build_dataset, cal_psnr, cal_ssim
from PIL import Image
from einops import rearrange
from torch.cuda.amp import autocast
import os
def warm_up_gpu(model, device):
    """
    Warm up the GPU by running a few dummy forward passes.
    Args:
        model: The PyTorch model to warm up.
        device: The device (e.g., CUDA or CPU).
    """
    input_shape_L = (1, 1, 250, 250)
    input_shape_R = (1, 3, 250, 250)
    dummy_input_L = torch.randn(input_shape_L).to(device).half()
    dummy_input_R = torch.randn(input_shape_R).to(device).half()

    for _ in range(50):
        with torch.no_grad():
            model(dummy_input_L, dummy_input_R)

    print("GPU warm-up complete.")

def save(img, path, flag, mode):
    img = rearrange(img, 'b C H W -> b H W C')
    img = torch.clamp(img.squeeze(), min=0, max=1)
    img = (img.cpu().numpy() * 255.0).round()
    img = Image.fromarray(np.uint8(img), mode=mode)
    cnt = path.split("\\")[-1].replace("L", flag)
    img.save(f"{cnt}")

def run(args, ckp):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize and load the model
    model = SICNet.SICNet(channels=args.MODEL.channels)
    checkpoint = torch.load(f"{ckp}")
    net = checkpoint['model']
    net = {key.replace("module.", ""): val for key, val in net.items()}
    model.load_state_dict(net)
    model.to(device).half()  # Convert model to FP16

    print("Warming up GPU...")
    warm_up_gpu(model, device)

    dataset_test = build_dataset(args, mode='test')
    n = len(dataset_test)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = DataLoader(dataset_test, batch_size=1, sampler=sampler_test, num_workers=4, pin_memory=True)

    start = time.time()
    psnr = 0
    ssim = 0
    valid_psnr_count = 0
    total_time_per_frame = []
    n=0

    with torch.no_grad(), autocast():
        model.eval()
        for idx, samples in enumerate(dataloader_val):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            print("Processing frame:", idx + 1)

            L = samples[0]['Left'].cuda().half()
            R = samples[0]['Right'].cuda().half()
            T = samples[0]['True'].cuda().half()
            TR = samples[0]['TR'].cuda().half()

            start_event.record()
            left_pre, right_pre, M, output = model(left=L, right=R)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)
            total_time_per_frame.append(elapsed_time)

            psnr_value = cal_psnr(output, T)
            ssim += cal_ssim(output, T)
            n+=1

            if np.isfinite(psnr_value):
                psnr += psnr_value
                valid_psnr_count += 1
            save(output, samples[1][0], "out", "RGB")

            log_message = f"Frame {idx + 1} inference time: {elapsed_time:.4f} ms with PSNR: {psnr_value:.4f} and SSIM: {cal_ssim(output, T):.4f}\n"
            
            print(log_message)

            with open(os.path.join("inference_log.txt"), "a") as log_file:
                log_file.write(log_message)

    end = time.time()
    avg_frame_time = sum(total_time_per_frame) / len(total_time_per_frame)

    print(f"Average total processing time: {(end - start) * 1000 / n:.4f} ms")
    print(f"Average CUDA inference time per frame: {avg_frame_time:.4f} ms")
    print(f"PSNR: {psnr / valid_psnr_count:.4f}")
    print(f"SSIM: {ssim / n:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config", default=1, type=int)
    parser.add_argument("--ckp")
    args = parser.parse_args()

    if args.config == 1:
        cfg = config.get_config()
    run(cfg, args.ckp)