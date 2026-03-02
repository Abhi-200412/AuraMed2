import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---------------------------
# PATHS
# ---------------------------
INPUT_ROOT = r"D:\Project\dataset\train"
OUTPUT_ROOT = r"D:\Project\aura-med\data\train"

IMAGE_SIZE = 224

# ---------------------------
# PREPROCESSING (NO AUGMENTATION)
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),                       # 0–255 → 0–1
    transforms.Normalize(mean=[0.5, 0.5, 0.5],   # → [-1, 1]
                         std=[0.5, 0.5, 0.5])
])

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def save_image(tensor, save_path):
    """
    Converts normalized tensor back to image and saves
    """
    tensor = (tensor * 0.5) + 0.5      # [-1,1] → [0,1]
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(save_path)

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for img_name in tqdm(images, desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(output_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image)
        save_image(tensor, save_path)

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    """
    Process ONLY training data.
    Preserves existing subfolders (e.g., normal, anomaly).
    """

    subfolders = [
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    for folder in subfolders:
        input_dir = os.path.join(INPUT_ROOT, folder)
        output_dir = os.path.join(OUTPUT_ROOT, folder)
        process_folder(input_dir, output_dir)

    print("\n✅ Training data preprocessing complete!")
    print(f"📁 Output saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()