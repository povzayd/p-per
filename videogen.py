'''video gen used old code from videogen9.py && adding object motion detection face 
highlighting bluring & --skip-ocr tag to reduce time '''
import os
import zipfile
import tempfile
import shutil
import cv2
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
import pytesseract
import csv
from tqdm import tqdm
#extract the zip func
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
#fetch img from folder
def get_image_files(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                image_files.append(os.path.join(root, f))
    image_files.sort()
    return image_files
##check for file intigrity
def validate_images(image_files):
    valid_images = []
    for img_path in tqdm(image_files, desc="🔎 Validating images"):
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_images.append(img_path)
        except Exception as e:
            print(f"⚠ Corrupted image skipped: {img_path} ({e})")
    return valid_images
#metadata/ time--stapms
def extract_timestamp(img_path):
    try:
        ts = os.path.getmtime(img_path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Unknown"
#ocr
def preprocess_for_ocr(pil_img):
    img_gray = pil_img.convert('L')
    img_np = np.array(img_gray)
    _, img_thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img_thresh)
#meta again
def log_metadata_and_ocr(image_files, log_path):
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Timestamp', 'Size (bytes)', 'OCR Text'])
        for img_path in tqdm(image_files, desc="📝 OCR + Logging"):
            timestamp = extract_timestamp(img_path)
            size = os.path.getsize(img_path)
            text = "Unreadable"
            try:
                img = Image.open(img_path).convert('RGB')
                ocr_img = preprocess_for_ocr(img)
                text = pytesseract.image_to_string(ocr_img).strip()
            except Exception as e:
                print(f"[❌ OCR Error] {img_path}: {e}")
            writer.writerow([os.path.basename(img_path), timestamp, size, text])
#anon-face
def anonymize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y:y+h, x:x+w] = blur
    return frame
#detect_faces fn
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame
#motion_highlight
def highlight_motion(prev_frame, curr_frame):
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return curr_frame, thresh
#heatmap generation
def generate_motion_heatmap(accum, output_path, preview=False):
    accum = np.clip(accum, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(accum, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap)
    print(f"🔥 Motion heatmap saved: {output_path}")

    if preview:
        cv2.imshow("🔥 Motion Heatmap Preview", heatmap)
        print("🖼 Press any key to close heatmap preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Close the window after key press
        print("✅ Heatmap preview closed.")
#compiling/video making
def make_video_from_images(image_files, output_path, fps=24, resolution=None,
                           enable_face=None, enable_motion=False,
                           motion_heatmap_path=None, preview_heatmap=False):
    if not image_files:
        print("🚫 No valid images to process.")
        return

    # Get first valid image for size
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                first_image = img.convert('RGB')
                default_width, default_height = first_image.size
                break
        except Exception:
            continue
    else:
        print("🚫 No valid images for resolution.")
        return

    if resolution:
        try:
            width, height = map(int, resolution.lower().split('x'))
        except Exception:
            print("❌ Invalid resolution format.")
            return
    else:
        width, height = default_width, default_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_frame = None
    motion_accum = None

    for img_path in tqdm(image_files, desc="🎞 Creating video"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((width, height))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            if enable_face == "anon":
                frame = anonymize_faces(frame)

            if enable_face == "detect":
                frame = detect_faces(frame)

            if enable_motion and prev_frame is not None:
                frame, motion_mask = highlight_motion(prev_frame, frame)
                if motion_accum is None:
                    motion_accum = np.zeros_like(motion_mask, dtype=np.float32)
                motion_accum += motion_mask.astype(np.float32)

            timestamp = extract_timestamp(img_path)
            cv2.putText(frame, timestamp, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            video.write(frame)
            prev_frame = frame.copy()
        except Exception as e:
            print(f"⚠ Skipping: {img_path} ({e})")
#call video release
    video.release()
    print(f"✅ Video saved: {output_path}")

    if enable_motion and motion_accum is not None and motion_heatmap_path:
        generate_motion_heatmap(motion_accum, motion_heatmap_path, preview=preview_heatmap)
#define tags &other things {help type}
def main():
    parser = argparse.ArgumentParser(description="Forensic image-to-video converter")
    parser.add_argument("input_path", help="Folder or ZIP with images")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR and metadata")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS")
    parser.add_argument("--out", type=str, default=".", help="Output folder")
    parser.add_argument("--resolution", type=str, help="Video resolution e.g. 1280x720")
    parser.add_argument("--face-anon", action="store_true", help="Blur detected faces")
    parser.add_argument("--face-detection", action="store_true", help="Detect and highlight faces")
    parser.add_argument("--motion-highlight", action="store_true", help="Highlight motion")
    parser.add_argument("--preview-heatmap", action="store_true", help="Show heatmap preview after rendering")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.out)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        folder_name = os.path.basename(os.path.normpath(input_path))
        folder = input_path
    elif zipfile.is_zipfile(input_path):
        folder_name = os.path.splitext(os.path.basename(input_path))[0]
        temp_dir = tempfile.mkdtemp()
        print("🗜 Extracting ZIP...")
        extract_zip(input_path, temp_dir)
        folder = temp_dir
    else:
        print("🚫 Invalid input. Must be folder or ZIP.")
        return

    output_video = os.path.join(output_dir, f"{folder_name}.mp4")
    output_csv = os.path.join(output_dir, f"{folder_name}_log.csv")
    heatmap_img = os.path.join(output_dir, f"{folder_name}_heatmap.jpg")

    print("🔍 Gathering images...")
    image_files = get_image_files(folder)
    image_files = validate_images(image_files)

    if not args.skip_ocr:
        log_metadata_and_ocr(image_files, output_csv)
        print(f"📄 Metadata saved: {output_csv}")
    else:
        print("🧾 OCR skipped.")

    enable_face = None
    if args.face_anon:
        enable_face = "anon"
    elif args.face_detection:
        enable_face = "detect"

    print("🎞 Generating video...")
    make_video_from_images(
        image_files,
        output_video,
        fps=args.fps,
        resolution=args.resolution,
        enable_face=enable_face,
        enable_motion=args.motion_highlight,
        motion_heatmap_path=heatmap_img if args.motion_highlight else None,
        preview_heatmap=args.preview_heatmap
    )

    if zipfile.is_zipfile(input_path):
        shutil.rmtree(folder)
        print("🧹 Cleaned up temp files.")

    # Gracefully exit OpenCV windows
    cv2.destroyAllWindows()  # Ensure all windows are closed

if __name__ == "__main__":
    main()
                                                                     
