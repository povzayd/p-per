---

Forensic Imaging and Video Reconstruction Toolkit aka p-per (peeper👀)

This toolkit is a unified suite of Python-based tools engineered for extracting visual evidence from raw binary files, PCAP captures, memory dumps, and more. It is ideal for digital forensic analysts, red/blue teamers, malware analysts, or CTF participants working on uncovering embedded image data and converting it into meaningful, analyzable video intelligence.

The toolkit is modular, terminal-friendly, and supports AI-powered image classification, metadata logging, face anonymization, and motion analysis.

---
Installation:
1. Clone The Repo:
```
git clone https://github.com/povzayd/p-per.git
```
2. Change Your Directory:
```
cd p-per
```
3. Activate Virtual Environment [If needed]
```
python3 venv venv && source venv/bin/activate 
```
4. Install Requirements:
```
pip3 install -r requirements.txt
``` 
5. Run The Required Tool.
6. `pcap2img.py` & `videogen.py` are lightweight & don't use alot of resources. On the
7. other hand `pcap2imgv2.py` utilizes alot of resources [In my case] :).
---
MODULE 1: JPEG Extraction from Binary Files – pcap2img.py

Purpose:

This script performs a deep scan of binary or media files to recover embedded JPEG images. It works by identifying JPEG start (0xFFD8) and end (0xFFD9) markers in the binary data using regular expressions.

Use Cases:

Recovery of deleted or hidden image files from raw memory dumps

Extracting evidence from file dumps in malware or ransomware cases

CTF tasks involving steganography or binary analysis


Key Features:

Supports various binary and media formats: .bin, .dat, .raw, .mp4, .avi, .mov, .jpg, .jpeg

Extracted images are saved in a clean, structured output directory

Uses tqdm for visual progress bars

Leverages prompt_toolkit for enhanced command-line interaction

Logs all actions and handles exceptions gracefully


Requirements:
```
pip install tqdm prompt_toolkit
```
Usage:
```
python pcap2img.py
```
You will be prompted:
```
🔍 Enter your file name:
```
Output:

A folder named after the input file (without extension), containing all frame_XXXX.jpg files

Each image is saved sequentially based on its offset in the binary


Example Run:
```
$ python jpeg_extractor.py
🔍 Enter your file name: disk_dump.bin
📂 Output directory created: 'disk_dump'
📸 Extracting images: 100%|████████████████████| 8/8
✅ Success! Extracted 8 JPEG images to 'disk_dump'
```
Limitations:

Does not validate JPEG file structure beyond the header and footer signatures

May produce corrupted or incomplete files if JPEGs are fragmented



---

MODULE 2: Advanced JPEG Extraction + AI Classification – pcap2imgv2.py

Purpose:

This is an enhanced version of pcap2img.py, supporting high-performance chunked reading for large files and AI-based classification of the extracted images using pretrained MobileNetV2.

Ideal For:

Analysts dealing with multi-gigabyte PCAPs or dumps

Automatic triage and tagging of image types

Prioritizing extracted data using AI prediction


Key Features:

Chunked binary parsing (1MB blocks) to reduce memory usage

Optional AI classification (--classify) using MobileNetV2 from PyTorch’s torchvision.models

Optional ZIP compression of output (--zip)

Clean CLI interface with argparse and logging

Customizable output directory and structured reporting


Requirements:
```
pip install torch torchvision pillow tqdm
```
Optional: Use a CUDA-enabled PyTorch build for GPU acceleration.

Usage:
```
python3 pcap2imgv2.py -i <input_file> [-o <output_dir>] [--zip] [--classify]
```
Command-Line Arguments:
```
Argument        Description

-i, --input     Path to the input binary (required)
-o, --output    Output directory (default: <input>_frames)
--zip   Compress the output folder into a .zip file
--classify      Run AI classification on extracted images
```

Example Runs:
```
python3 pcap2imgv2.py -i traffic.pcap
```
```
python3 pcap2imgv2.py -i dump.bin --zip
```
```
python3 pcap2imgv2.py -i memory_dump.raw --classify
```
```
python3 pcap2imgv2.py -i data.pcap -o ./output_frames
```
---
AI Classification Output:
```
🧠 frame_0021.jpg → "laptop"
🧠 frame_0022.jpg → "zebra"
```
Performance Logging:
```
Total number of images extracted

Overall data size extracted

Elapsed processing time

```
Limitations:

Only supports JPEG recovery (FFD8 to FFD9)

ImageNet-trained classifier may mislabel specialized forensics content

Damaged frames may fail classification



---

MODULE 3: Forensic Video Generator – videogen.py

Purpose:

Converts a sequence of images (from folder or .zip) into a forensic-grade MP4 video with optional overlays including timestamps, face anonymization, OCR metadata, motion detection, and heatmap generation.

Use Cases:

Reconstructing surveillance sequences from recovered frames

Creating visual timelines from malware dump images

Anonymizing human faces in sensitive image sequences


Key Features:

Converts images from folders or .zip archives into videos

Timestamps auto-injected based on image metadata

Gaussian blur for anonymizing faces

Bounding boxes around detected faces using Haar cascades

Motion detection with contour highlights

Optional heatmap visualization of detected motion

OCR-powered text extraction from frames and metadata logging


Requirements:
```
pip install opencv-python-headless pillow numpy pytesseract tqdm
```
Usage:
```
python videogen.py <input_path> [OPTIONS]
```
Supported CLI Options:
```
Flag    Description

--skip-ocr      Skips OCR scanning for faster video generation
--fps <int>     Set frames per second (default: 24)
--out <folder>  Set custom output directory
--resolution WxH        Custom resolution (e.g., 1280x720)
--face-anon     Anonymize faces with Gaussian blur
--face-detection        Draw face detection boxes
--motion-highlight      Highlight detected motion between frames
--preview-heatmap       Show heatmap preview at the end

```
Output Files:
```
*.mp4 – Reconstructed forensic video

*_log.csv – Metadata and OCR results (if enabled)

*_heatmap.jpg – Motion heatmap visualization

```
Example Commands:
```
python videogen.py ./frames
```
```
python videogen.py ./surveillance.zip --face-anon --motion-highlight --preview-heatmap
```
```
python videogen.py ./evidence --skip-ocr
```
```
python videogen.py ./input_folder --resolution 1280x720 --out ./results
```
Limitations:

Relies on consistent timestamp metadata for sequencing

Motion detection is basic and may flag minor shifts

OCR and face detection can add processing overhead



---

Summary
```
Tool    Purpose Core Tech       Highlights

pcap2img.py     Simple JPEG extractor   Regex + CLI     Prompt-driven, fast, lightweight
pcap2imgv2.py   Advanced extractor + AI classifier      PyTorch, Argparse  Classify & zip support
videogen.py     Image-to-video forensic builder OpenCV, Tesseract Anonymization, motion, metadata
```


---

>Made With 🪣                              
>Special Thanks To [@una55](https://github.com/una55)              
>Special Thanks To [@xbee9](https://github.com/xbee9)
