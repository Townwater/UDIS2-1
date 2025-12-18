import os
from PIL import Image

root_dir = r"D:\UDIS2\sorted_image_testing_set"
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

for scene in os.listdir(root_dir):
    scene_path = os.path.join(root_dir, scene)
    if not os.path.isdir(scene_path):
        continue

    images = [f for f in os.listdir(scene_path)
              if os.path.splitext(f)[1].lower() in valid_exts]
    images.sort()

    # === 第一輪：先全部改成暫存名稱，避免名稱衝突 ===
    for i, img_name in enumerate(images):
        src_path = os.path.join(scene_path, img_name)
        temp_name = f"temp_{i:04d}{os.path.splitext(img_name)[1].lower()}"
        temp_path = os.path.join(scene_path, temp_name)
        os.rename(src_path, temp_path)

    # === 第二輪：再統一改成 0001.jpg 格式 ===
    temp_images = [f for f in os.listdir(scene_path) if f.startswith("temp_")]
    temp_images.sort()
    for i, img_name in enumerate(temp_images):
        src_path = os.path.join(scene_path, img_name)
        new_name = f"{i:04d}.jpg"
        dst_path = os.path.join(scene_path, new_name)

        # 若原副檔名非 .jpg，則轉檔
        if not img_name.lower().endswith('.jpg'):
            im = Image.open(src_path).convert("RGB")
            im.save(dst_path)
            os.remove(src_path)
        else:
            os.rename(src_path, dst_path)

    print(f"[✓] Renamed {len(images)} images in {scene}")