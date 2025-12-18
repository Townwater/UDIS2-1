import os
import itertools
import random
import pandas as pd

root = r"D:\UDIS2\sorted_image_training_set"
scenes = sorted(os.listdir(root))

pairs = []

for i, scene_i in enumerate(scenes):
    imgs_i = sorted(os.listdir(os.path.join(root, scene_i)))

    # ✅ 同場景內 pair (正樣本)
    pos_pairs = list(itertools.combinations(imgs_i, 2))
    for a, b in pos_pairs:
        pairs.append([f"{scene_i}/{a}", f"{scene_i}/{b}", 1])

    # ❌ 不同場景 pair (負樣本) — SAME COUNT AS POS
    num_pos = len(pos_pairs)
    for _ in range(num_pos):
        other_scene = random.choice([s for s in scenes if s != scene_i])
        imgs_j = os.listdir(os.path.join(root, other_scene))
        a = random.choice(imgs_i)
        b = random.choice(imgs_j)
        pairs.append([f"{scene_i}/{a}", f"{other_scene}/{b}", 0])

df = pd.DataFrame(pairs, columns=["image1", "image2", "label"])
df.to_csv(r"D:\UDIS2\pairs.csv", index=False)
print(f"✅ 生成 {len(pairs)} 筆 pair，儲存為 dataset/pairs.csv")