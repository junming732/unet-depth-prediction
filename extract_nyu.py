import h5py
import numpy as np
from PIL import Image
import os

# Configuration
mat_file = '/home/junming/nobackup_junming/nyu_depth_v2/nyu_depth_v2_labeled.mat'
output_dir = '/home/junming/nobackup_junming/nyu_depth_v2/extracted_data'

# Create directories
os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)

print(f"Reading {mat_file}...")
with h5py.File(mat_file, 'r') as f:
    # H5PY reads data as (N, C, W, H), we need to transpose to (H, W, C)
    images = f['images']
    depths = f['depths']

    num_images = images.shape[0]
    print(f"Extracting {num_images} images...")

    for i in range(num_images):
        # Extract RGB: Transpose from (3, 640, 480) -> (480, 640, 3)
        img = f['images'][i]
        img = np.transpose(img, (2, 1, 0)).astype('uint8')

        # Extract Depth: Transpose from (640, 480) -> (480, 640)
        # Note: Depth is in meters
        depth = f['depths'][i]
        depth = np.transpose(depth, (1, 0)).astype('float32')

        # Save files
        Image.fromarray(img).save(f"{output_dir}/rgb/{i:04d}.jpg")

        # Save depth as .png (uint16 mm) or raw .npy depending on your loader
        # Here we save as .npy to preserve float precision for benchmarking
        np.save(f"{output_dir}/depth/{i:04d}.npy", depth)

        if i % 100 == 0:
            print(f"Processed {i}/{num_images}")

print("Done!")