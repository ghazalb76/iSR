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
import lpips


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


def best_window(metrics, key, window=100, minimize=False, finite_only=False):
    """
    metrics: list of dicts with fields: name, psnr, ssim, time_ms
    key: 'psnr' | 'ssim' | 'time_ms'
    window: window length
    minimize: True -> choose smallest average (for time)
    finite_only: True -> ignore non-finite values in average (useful for PSNR)
    Returns: (best_start_idx, best_end_idx_inclusive, best_avg, names_in_window)
    """
    n = len(metrics)
    if n < window:
        return None

    best_avg = None
    best_i = None

    for i in range(0, n - window + 1):
        chunk = metrics[i:i + window]
        vals = [m[key] for m in chunk]

        if finite_only:
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                continue

        avg = float(np.mean(vals))

        if best_avg is None:
            best_avg = avg
            best_i = i
        else:
            if minimize:
                if avg < best_avg:
                    best_avg = avg
                    best_i = i
            else:
                if avg > best_avg:
                    best_avg = avg
                    best_i = i

    if best_i is None:
        return None

    names = [m["name"] for m in metrics[best_i:best_i + window]]
    return best_i, best_i + window - 1, best_avg, names


def run(args, ckp):
    lpips_model = lpips.LPIPS(net='alex')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lpips_model.to(device)

    # Initialize and load the model
    model = SICNet.SICNet(channels=args.MODEL.channels)
    checkpoint = torch.load(f"{ckp}", map_location="cpu")
    net = checkpoint['model']
    net = {key.replace("module.", ""): val for key, val in net.items()}
    model.load_state_dict(net)
    model.to(device).half()  # Convert model to FP16

    print("Warming up GPU...")
    warm_up_gpu(model, device)

    dataset_test = build_dataset(args, mode='test')
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = DataLoader(
        dataset_test,
        batch_size=1,
        sampler=sampler_test,
        num_workers=4,
        pin_memory=True
    )

    start = time.time()
    psnr = 0.0
    ssim = 0.0
    lpips_score = 0.0
    valid_psnr_count = 0
    total_time_per_frame = []
    n = 0

    # Collect per-frame metrics to find best 100 consecutive frames
    frame_metrics = []

    with torch.no_grad(), autocast():
        model.eval()
        for idx, samples in enumerate(dataloader_val):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            print("Processing frame:", idx + 1)

            L = samples[0]['Left'].cuda().half()
            R = samples[0]['Right'].cuda().half()
            T = samples[0]['True'].cuda().half()
            TR = samples[0]['TR'].cuda().half()  # kept, even if unused, because your dataset returns it

            start_event.record()
            left_pre, right_pre, M, output = model(left=L, right=R)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)
            total_time_per_frame.append(elapsed_time)

            psnr_value = cal_psnr(output, T)
            ssim_value = cal_ssim(output, T)

            # Ensure both 'output' and 'T' are on the same device as the LPIPS model
            output_lp = output.to(device)
            T_lp = T.to(device)

            # Compute LPIPS score
            lpips_value = lpips_model(output_lp, T_lp)
            lpips_score += lpips_value.item()

            n += 1

            if np.isfinite(psnr_value):
                psnr += psnr_value
                valid_psnr_count += 1

            ssim += ssim_value

            # Save output image
            save(output, samples[1][0], "out", "RGB")

            # Pick a name that identifies the input frame
            # samples[1][0] seems to be a string/path in your code
            img_name_raw = samples[1][0]
            if isinstance(img_name_raw, str):
                img_name = os.path.basename(img_name_raw)
            else:
                img_name = str(img_name_raw)

            # Store per-frame metrics
            frame_metrics.append({
                "name": img_name,
                "psnr": float(psnr_value) if psnr_value is not None else float("nan"),
                "ssim": float(ssim_value) if ssim_value is not None else float("nan"),
                "time_ms": float(elapsed_time),
            })

            log_message = (
                f"Frame {idx + 1} inference time: {elapsed_time:.4f} ms "
                f"with PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, "
                f"and LPIPS: {lpips_value.item():.4f}\n"
            )

            print(log_message)

            with open(os.path.join("inference_log.txt"), "a") as log_file:
                log_file.write(log_message)

    end = time.time()
    avg_frame_time = sum(total_time_per_frame) / len(total_time_per_frame) if total_time_per_frame else float("nan")

    print(f"Average total processing time: {(end - start) * 1000 / n:.4f} ms")
    print(f"Average CUDA inference time per frame: {avg_frame_time:.4f} ms")
    print(f"PSNR: {psnr / valid_psnr_count:.4f}" if valid_psnr_count > 0 else "PSNR: N/A (no finite PSNR values)")
    print(f"SSIM: {ssim / n:.4f}")
    print(f"LPIPS: {lpips_score / n:.4f}")

        # ===== Compute averages per non-overlapping 100-frame blocks =====
    BLOCK_SIZE = 100
    total_frames = len(frame_metrics)

    if total_frames < BLOCK_SIZE:
        print(f"\nNot enough frames for block size {BLOCK_SIZE}. Total frames: {total_frames}")
    else:
        num_blocks = total_frames // BLOCK_SIZE

        print(f"\n===== Averages per {BLOCK_SIZE}-Frame Block =====")

        for b in range(num_blocks):
            start_i = b * BLOCK_SIZE
            end_i = start_i + BLOCK_SIZE

            block = frame_metrics[start_i:end_i]

            # Extract values
            psnr_vals = [m["psnr"] for m in block if np.isfinite(m["psnr"])]
            ssim_vals = [m["ssim"] for m in block]
            time_vals = [m["time_ms"] for m in block]

            avg_psnr = np.mean(psnr_vals) if len(psnr_vals) > 0 else float("nan")
            avg_ssim = np.mean(ssim_vals)
            avg_time = np.mean(time_vals)

            print(f"\nBlock {b+1}: Frames {start_i+1} → {end_i}")
            print(f"Average PSNR : {avg_psnr:.6f}")
            print(f"Average SSIM : {avg_ssim:.6f}")
            print(f"Average Time : {avg_time:.6f} ms")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config", default=1, type=int)
    parser.add_argument("--ckp")
    args = parser.parse_args()

    if args.config == 1:
        cfg = config.get_config()
    else:
        cfg = config.get_config()

    run(cfg, args.ckp)
