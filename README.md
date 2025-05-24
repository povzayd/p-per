

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
