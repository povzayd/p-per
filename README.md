

STEP I
JPEG Extraction from Binary Files [pcap2img.py]

This script scans a binary or media file and extracts embedded JPEG images by identifying JPEG start (0xFFD8) and end (0xFFD9) markers using regular expressions.

Features

Supports `.bin`, `.dat`, `.raw`, `.mp4`, `.avi`, `.mov`, `.jpg`, `.jpeg` files.

Extracts all embedded JPEG images and saves them in a dedicated folder.

Uses `tqdm` for progress bars and `prompt_toolkit` for clean CLI interaction.

Built-in logging and error handling.


Requirements
```
Python 3.6+

tqdm

prompt_toolkit

```
Install dependencies using:
```
pip install tqdm prompt_toolkit
```
Usage

Run the script directly:
```
python pcap2img.py
```

You'll be prompted to enter a file name:
```

🔍 Enter your file name:
```

Provide the path to your .bin, .dat, .raw, or media file. If JPEG signatures are found, extracted images will be saved as frame_XXXX.jpg inside a folder named after the input file (excluding extension).

Example
```
$ python jpeg_extractor.py
🔍 Enter your file name: disk_dump.bin

📂 Output directory created: 'disk_dump'
📸 Extracting images: 100%|████████████████████| 8/8
✅ Success! Extracted 8 JPEG images to 'disk_dump'
```
Notes:

The script does not validate JPEG integrity beyond header/footer signatures.

Ideal for forensics, CTF challenges, or reverse engineering tasks.
---
---
> pcap2imgv2.py

A Python tool for extracting JPEG images from binary files (such as PCAP, raw memory dumps, etc.), with optional AI-powered image classification and zipped output support.

---

Features

Efficient chunked JPEG extraction from large binary files.

Optional AI classification of images using MobileNetV2 (ImageNet pretrained).

Automatically zips extracted images if required.

Informative logging and summary report.

Clean and modular design using `argparse`, `torch`, `tqdm`, and `Pillow`.

---

Dependencies

Make sure you have Python 3.7+ and the following packages installed:
```
pip install torch torchvision pillow tqdm
```

Optional but recommended: Install CUDA-compatible PyTorch if using GPU.


---

Usage

Command-line Interface
```

python3 pcap2img.py -i <input_file> [-o <output_dir>] [--zip] [--classify]
```
Arguments

Argument	Description
```
-i, --input	Path to the input binary file (e.g., .pcap, .bin, etc.). (Required)
-o, --output	Output directory to save images. Default is <input>_frames.
--zip	Zip the output directory after extraction.
--classify	Classify each extracted image using MobileNetV2 AI model.

```

---

Example Commands

Basic Extraction:
```
python3 pcap2img.py -i traffic.pcap
```
Extract and Zip:
```
python3 pcap2img.py -i dump.bin --zip
```
Extract with AI Classification:
```
python3 pcap2img.py -i memory_dump.raw --classify
```
Custom Output Directory:
```
python3 pcap2img.py -i data.pcap -o ./output_frames

```

---

AI Classification

When the `--classify` flag is used, each image is passed through a pre-trained MobileNetV2 model from PyTorch's torchvision.models. The top-1 prediction from ImageNet categories is logged alongside the saved image.

Example Output:
```

🧠 frame_0021.jpg → "laptop"
🧠 frame_0022.jpg → "zebra"

```
---

Logging and Reporting

The script provides real-time logs and ends with a performance summary:
```
Number of images extracted

Total size of extracted data

Elapsed processing time

```

---

Performance

Optimized for handling large files with minimal memory usage using 1MB chunks.


---

Output Structure

Example directory structure after extraction:
```
example_frames/
├── frame_0000.jpg
├── frame_0001.jpg
├── frame_0002.jpg
└── ...
```
If --zip is used:
```
example_frames.zip
```

---

Known Limitations

Only extracts JPEG images (FFD8...FFD9) from raw data.

Some corrupted frames may be extracted but fail AI classification.

The classification is based on ImageNet, which may not be relevant for certain contexts (e.g., surveillance footage, malware dumps).



---
---

