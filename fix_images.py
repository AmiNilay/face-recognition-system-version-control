"""
Image Normalization and Repair Utility

Scans the known_faces directory to find images with corrupted metadata,
incompatible channel depths (e.g. 4-channel RGBA, indexed color, grayscale),
or non-standard extensions, and re-encodes them into standardized 24-bit RGB JPEGs.

All diagnostic output uses Indian Standard Time (IST) logging without emojis.
"""

import os
from pathlib import Path
from PIL import Image
from src.utils import format_ist_timestamp

def fix_images(folder: str = "known_faces"):
    """
    Validate, normalize, and fix all images in the known_faces folder.
    Converts corrupted, alpha-channel, or non-RGB images into clean standard RGB JPEGs.

    Args:
        folder (str): Directory containing image files to sanitize.
    """
    if not os.path.exists(folder):
        print(f"[ERROR] Directory '{folder}' does not exist.")
        return

    print("=" * 50)
    print("           IMAGE NORMALIZATION UTILITY")
    print(f"            Executed: {format_ist_timestamp()}")
    print("=" * 50)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    processed_count = 0
    error_count = 0

    for filename in os.listdir(folder):
        if filename.lower() == "readme.md" or filename.startswith('.'):
            continue

        filepath = Path(folder) / filename
        if not filepath.is_file() or filepath.suffix.lower() not in valid_extensions:
            continue

        try:
            with Image.open(filepath) as img:
                # Convert RGBA, CMYK, Palette, or Grayscale images into 24-bit RGB
                rgb_img = img.convert("RGB")

                # Target standardized filename with .jpg extension
                target_filepath = filepath.with_suffix('.jpg')
                temp_filepath = filepath.with_name(f"{filepath.stem}_temp_fix.jpg")

                # Save clean, optimized JPEG with 95% quality
                rgb_img.save(temp_filepath, "JPEG", quality=95)

            # Atomically replace old file with newly formatted file
            if filepath.exists():
                os.remove(filepath)
            os.rename(temp_filepath, target_filepath)

            print(f"[OK] Normalized: {filename} -> {target_filepath.name}")
            processed_count += 1

        except Exception as err:
            print(f"[ERROR] Could not process {filename}: {err}")
            error_count += 1

    print(f"\n[INFO] Normalization complete. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    fix_images()
