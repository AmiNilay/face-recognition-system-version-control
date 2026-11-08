# 👤 Known Faces Directory

This directory stores reference images for registered persons recognized by the system.

---

## 📸 Image Guidelines

1. **Filename Format:**  
   Name the image file with the person's real name or identifier:
   * `John_Doe.jpg` ➔ Displayed as **John Doe**
   * `Alice_Smith.png` ➔ Displayed as **Alice Smith**
   * Avoid special characters like `/ \ : * ? " < > |`

2. **Supported Formats:**
   * `.jpg` / `.jpeg` (Recommended)
   * `.png`
   * `.webp`
   * `.bmp`
   * `.tiff`

3. **Photo Quality Best Practices:**
   * **Single Face:** Ensure only **one** person's face is clearly visible per photo.
   * **Lighting:** Well-lit, front-facing, neutral expression.
   * **Obstructions:** Remove sunglasses, masks, or extreme hats. Plain reading glasses are acceptable if worn consistently.
   * **Resolution:** Minimum $300 \times 300$ pixels (recommended $600 \times 600$ to $1200 \times 1200$).

---

## ⚡ How to Add Faces

### Option 1: Via the GUI Application
1. Launch the application: `run_gui.bat`
2. Click **➕ Add** under the Known Faces panel to select an image from your computer.
3. Or start the camera and click **📸 Capture Face** to take a live photo.

### Option 2: Via CLI Script
```bash
python add_faces.py
```
Follow the interactive prompt to position your face and press **SPACE** to capture.

### Option 3: Manual Drop
Place your `.jpg` or `.png` file directly into this `known_faces/` folder, then click **🔄 Refresh** in the GUI or restart the application. Face encodings will automatically be generated and saved.
