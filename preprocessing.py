import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---------------------------
# PATHS
# ---------------------------
INPUT_ROOT = r"D:\Project\dataset"
OUTPUT_ROOT = r"D:\Project\aura-med\data"

IMAGE_SIZE = 224

# ---------------------------
# PREPROCESS PIPELINE
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # ensure correct size
    transforms.ToTensor(),                         # convert to tensor [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],          # normalize → [-1,1]
                         [0.5, 0.5, 0.5])
])

# ---------------------------
# SAVE FUNCTION
# ---------------------------
def save_image(tensor, path):
    """
    Convert normalized tensor back to image and save.
    """
    tensor = (tensor * 0.5) + 0.5       # convert [-1,1] → [0,1]
    tensor = tensor.clamp(0, 1)

    image = transforms.ToPILImage()(tensor)
    image.save(path)

# ---------------------------
# PROCESS SINGLE DIRECTORY
# ---------------------------
def process_directory(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    images = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_name in tqdm(images, desc=f"Processing {input_dir}"):

        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        image = Image.open(input_path).convert("RGB")

        tensor = preprocess(image)

        save_image(tensor, output_path)

# ---------------------------
# WALK THROUGH DATASET
# ---------------------------
def main():

    for root, dirs, files in os.walk(INPUT_ROOT):

        # skip empty folders
        if len(files) == 0:
            continue

        rel_path = os.path.relpath(root, INPUT_ROOT)
        output_dir = os.path.join(OUTPUT_ROOT, rel_path)

        process_directory(root, output_dir)

    print("\n✅ Preprocessing complete!")
    print(f"Processed dataset saved to: {OUTPUT_ROOT}")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    main()