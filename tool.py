####modified version prolly 2.0 of pcap2img.py
import os
import re
import logging
import argparse
import time
import shutil

from tqdm import tqdm
from PIL import Image
from io import BytesIO

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# AI model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with updated weights syntax
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).to(device).eval()

# Load ImageNet class labels (these are also available from the weights)
imagenet_classes = weights.meta["categories"]

# Use transforms from weights for consistency
transform = weights.transforms()

def classify_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        _, predicted = output.max(1)
        return imagenet_classes[predicted.item()]
    except Exception as e:
        logger.warning(f"⚠ Classification failed: {e}")
        return "Unclassified"
###process in put file
def extract_jpegs_chunked(file_path, output_dir, zip_output=False, classify=False):
    chunk_size = 1024 * 1024  # 1MB
    buffer = b""
    frame_count = 0
    image_sizes = []
    predictions = []

    jpeg_start = b'\xff\xd8'
    jpeg_end = b'\xff\xd9'

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📂 Output directory: {output_dir}")

    start_time = time.time()

    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            buffer += chunk
            while True:
                start = buffer.find(jpeg_start)
                end = buffer.find(jpeg_end, start + 2)
                if start != -1 and end != -1:
                    jpeg_data = buffer[start:end+2]
                    buffer = buffer[end+2:]
'''
Now Write output....add zip option also for videogen.py

'''
                    output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                    with open(output_file, 'wb') as out:
                        out.write(jpeg_data)
                    image_sizes.append(len(jpeg_data))

                    if classify:
                        label = classify_image(jpeg_data)
                        predictions.append((output_file, label))
                        logger.info(f"🧠 {output_file} → {label}")

                    frame_count += 1
                else:
                    break

    elapsed = time.time() - start_time
    logger.info(f"\n📊 Summary Report:")
    logger.info(f" - Extracted: {frame_count} images")
    logger.info(f" - Total size: {sum(image_sizes) / 1024:.2f} KB")
    logger.info(f" - Time taken: {elapsed:.2f} seconds")

    if zip_output:
        zip_name = shutil.make_archive(output_dir, 'zip', output_dir)
        logger.info(f"🗜 Output zipped to: {zip_name}")

    return frame_count, image_sizes, predictions
##extraction part..........
#import turtle
#import torch
def main():
    parser = argparse.ArgumentParser(description="Extract embedded JPEGs from a binary file.")
    parser.add_argument("-i", "--input", required=True, help="Input binary file")
    parser.add_argument("-o", "--output", help="Output directory (default: <input>_frames)")
    parser.add_argument("--zip", action="store_true", help="Zip the output folder")
    parser.add_argument("--classify", action="store_true", help="Classify each image with AI")
    args = parser.parse_args()

    input_file = args.input
    if not os.path.isfile(input_file):
        logger.error(f"🚫 File not found: {input_file}")
        return

    output_dir = args.output if args.output else f"{os.path.splitext(input_file)[0]}_frames"

    extract_jpegs_chunked(input_file, output_dir, zip_output=args.zip, classify=args.classify)

if __name__ == "__main__":
    main()
