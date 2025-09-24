import torch
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from utils import reparameterize, tensor2uint
import os
from model import LitRT4KSR_Rep
import config_ft
import cv2  # Import OpenCV
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision.transforms import ToTensor, Resize
import lpips  # Import LPIPS

def main():
    psnr_metric = PeakSignalNoiseRatio()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)  # Assuming images are normalized to [0, 1]
    lpips_metric = lpips.LPIPS(net='alex')
    to_tensor = ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    litmodel = LitRT4KSR_Rep.load_from_checkpoint(checkpoint_path=config_ft.checkpoint_path_infer, config=config_ft)
    if config_ft.infer_reparameterize:  # reparameterize model
        litmodel.model = reparameterize(config_ft, litmodel.model, device, save_rep_checkpoint=False)
        # Assuming your model is called 'model'
   
    litmodel.to(device)
    litmodel.eval()
    hidden_state = None
    # Print the number of parameters
    model_parameters = filter(lambda p: p.requires_grad, litmodel.model.parameters())
    params_count = sum([p.numel() for p in model_parameters])
    print(f"Number of parameters in the model: {params_count}")
    # Warm-up
    for _ in range(5):
        dummy_input = torch.rand(1, 3, 200, 400, device=device)
        with torch.no_grad():
            litmodel(dummy_input)

    inference_times = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    save_path = config_ft.infer_save_path
    ground_truth_path = config_ft.ground_truth_path

    for filename in os.listdir(config_ft.infer_lr_image_path):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):  # skip non-image files
            continue

        img_path = os.path.join(config_ft.infer_lr_image_path, filename)
        gt_img_path = os.path.join(ground_truth_path, filename)
        lr_image = Image.open(img_path).convert("RGB")
        gt_image = Image.open(gt_img_path).convert("RGB")
        lr_sample = TF.to_tensor(np.array(lr_image) / 255.0).unsqueeze(0).to(device)

        # Measure inference
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        lr_sample = lr_sample.float()
        start_event.record()
        with torch.no_grad():
            sr_sample, hidden_state = litmodel.predict_step(lr_sample, hidden_state)
        end_event.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        inference_time = start_event.elapsed_time(end_event)  # milliseconds
        inference_times.append(inference_time)

        # Convert tensor to image and save
        sr_sample = tensor2uint(sr_sample * 255.0)
        image_sr_PIL = Image.fromarray(sr_sample)

        sr_sample_cv = cv2.cvtColor(np.array(image_sr_PIL), cv2.COLOR_RGB2BGR)

        base, ext = os.path.splitext(filename)
        image_name = f"{base}{ext}"  # e.g., "image_sr.png" if original was "image.png"

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, image_name), sr_sample_cv)

        

        gt_image = to_tensor(gt_image).unsqueeze(0)
        sr_sample = to_tensor(sr_sample).unsqueeze(0)

        psnr = psnr_metric(gt_image, sr_sample).item()
        ssim = ssim_metric(gt_image, sr_sample).item()
        lpips_score = lpips_metric(gt_image, sr_sample).item()

        log_message = f"Inference done in {inference_time:.3f} ms for image {filename} with PSNR: {psnr}, SSIM: {ssim:.4f}, LPIPS: {lpips_score:.4f}\n"
        print(log_message)
        with open(os.path.join(save_path, "inference_log.txt"), "a") as log_file:
            log_file.write(log_message)

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_score)


    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_lpips = sum(lpips_values) / len(lpips_values)
    print(f"Average inference time: {avg_inference_time:.3f} ms")
    print(f"Average PSNR: {avg_psnr:.3f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")



if __name__ == "__main__":
    main()