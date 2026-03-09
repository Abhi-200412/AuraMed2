import os
import random
import shutil

SOURCE = "Dataset (2) new"
DEST = "dataset"

# Source folders
train_normal = os.path.join(SOURCE, "train", "normal")
train_anomaly = os.path.join(SOURCE, "train", "anomaly")
test_normal = os.path.join(SOURCE, "test", "normal")
test_anomaly = os.path.join(SOURCE, "test", "anomaly")

# Destination folders
train_normal_out = os.path.join(DEST, "train", "normal")
test_normal_out = os.path.join(DEST, "test", "normal")
test_anomaly_out = os.path.join(DEST, "test", "anomaly")

# Create folders
for path in [train_normal_out, test_normal_out, test_anomaly_out]:
    os.makedirs(path, exist_ok=True)


# Collect images safely
def collect(folder):
    files = []
    
    if not os.path.exists(folder):
        print(f"⚠ Folder not found: {folder}")
        return files

    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            files.append(path)

    return files


# Combine all NORMAL images
normal_files = collect(train_normal) + collect(test_normal)

# Combine all ANOMALY images
anomaly_files = collect(train_anomaly) + collect(test_anomaly)

print("Normal images:", len(normal_files))
print("Anomaly images:", len(anomaly_files))


# Shuffle
random.shuffle(normal_files)
random.shuffle(anomaly_files)


# Split NORMAL images into train/test
split_index = int(len(normal_files) * 0.8)

train_normals = normal_files[:split_index]
test_normals = normal_files[split_index:]


# Copy helper
def copy(files, dest):
    for f in files:
        shutil.copy2(f, os.path.join(dest, os.path.basename(f)))


# Copy files
copy(train_normals, train_normal_out)
copy(test_normals, test_normal_out)
copy(anomaly_files, test_anomaly_out)


print("✅ Dataset successfully prepared for anomaly detection")