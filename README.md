# iSR

## Colorization Module

### Run Inference
Use the following command to run the colorization model:

```bash
python testquantization.py --config 1 --ckp colorization/checkpoints/BeatSaber/scale8Channels48.pth
```

### Prepare for Stitching

1. Take the generated _out.png images and the corresponding _HR.png images.

2. Place them into two separate folders named:
L/
R/

3. Update the paths to these folders inside stitch.py.

#### Run Stitching

After updating the paths, run:

```bash
python stitch.py
```


## Super-Resolution Module

### Configure Paths

Open config_ft.py and update the following:

infer_lr_image_path → path to the low-resolution images

infer_save_path → output folder where results will be saved

ground_truth_path → path to the ground-truth HR images (available in OneDrive)

### Run Inference

```bash
python infer.py
```