import re
import os
import logging
from prompt_toolkit import PromptSession
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def is_supported_file(file_path):
    supported_exts = ('.bin', '.dat', '.raw', '.mp4', '.avi', '.mov', '.jpg', '.jpeg')
    return file_path.lower().endswith(supported_exts)

def extract_jpeg_images(file_path):
    if not is_supported_file(file_path):
        logger.warning(f"⚠ Warning: File extension may not contain JPEGs: {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        logger.error(f"🚫 Error: File '{file_path}' not found.")
        return
    except Exception as e:
        logger.error(f"⚠ Error reading file: {e}")
        return

    logger.info(f"🔍 Scanning '{file_path}' for JPEG images...")

    # Improved JPEG pattern (start with 0xFFD8 and end with 0xFFD9)
    jpeg_pattern = re.compile(b'\xff\xd8(?:.|\n)*?\xff\xd9')
    matches = jpeg_pattern.findall(data)

    if not matches:
        logger.info("❌ No JPEG images found in the file.")
        return

    output_dir = os.path.splitext(file_path)[0]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📂 Output directory created: '{output_dir}'")

    for i, jpeg in enumerate(tqdm(matches, desc="📸 Extracting images")):
        output_file = os.path.join(output_dir, f'frame_{i:04d}.jpg')
        with open(output_file, 'wb') as out:
            out.write(jpeg)

    logger.info(f"✅ Success! Extracted {len(matches)} JPEG images to '{output_dir}'")

def main():
    session = PromptSession("🔍 Enter your file name: ")
    try:
        file_path = session.prompt()
        extract_jpeg_images(file_path)
    except KeyboardInterrupt:
        logger.warning("\n🚪 Operation cancelled by user.")

if __name__ == "__main__":
    main()
                                                 
