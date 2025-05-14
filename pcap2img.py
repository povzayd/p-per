import re
import os
from prompt_toolkit import PromptSession

def extract_jpeg_images(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"🚫 Error: File '{file_path}' not found.")
        return
    except Exception as e:
        print(f"⚠️ Error reading file: {e}")
        return

    jpeg_pattern = re.compile(b'\xff\xd8.*?\xff\xd9', re.DOTALL)
    matches = jpeg_pattern.findall(data)

    output_dir = os.path.splitext(file_path)[0]
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Created output directory: '{output_dir}'")

    for i, jpeg in enumerate(matches):
        output_file = os.path.join(output_dir, f'frame_{i:04d}.jpg')
        with open(output_file, 'wb') as out:
            out.write(jpeg)

    print(f"✅ Success! Extracted {len(matches)} JPEG images 📸 to '{output_dir}'")

def main():
    session = PromptSession("🔍 Enter your file name: ")
    file_path = session.prompt()
    extract_jpeg_images(file_path)

if __name__ == "__main__":
    main()
