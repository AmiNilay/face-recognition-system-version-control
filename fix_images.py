import cv2
import os

def fix_images():
    """Fix all images in known_faces folder"""
    folder = "known_faces"
    
    if not os.path.exists(folder):
        print("No known_faces folder found")
        return
    
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(folder, filename)
            

            img = cv2.imread(filepath)
            if img is not None:

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                

                new_path = filepath.replace('.jpg', '_fixed.jpg')
                cv2.imwrite(new_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                

                os.remove(filepath)
                os.rename(new_path, filepath)
                print(f"✅ Fixed: {filename}")
            else:
                print(f"❌ Could not read: {filename}")

                os.remove(filepath)
                print(f"🗑️  Deleted corrupted: {filename}")
    
    print("\n✅ All images fixed!")
    print("Now re-add your faces using: python add_faces.py")

if __name__ == "__main__":
    fix_images()