import sys
import os
import pandas as pd
import numpy as np
import json

MODEL_DIR = os.path.join('./..','models')
CSV_DIR = os.path.join('..','csv')
IMG_DIR = 'images'

PROMPT = "Use given reference image and other 4 images to understand the scenario and give a logical sequence which the images are following like list ex. [2,1,4,3]. Also give the a one line reason for that. Do not give any output other that sequence and reason."


from itertools import combinations

def pairwise_accuracy(predicted, true_order):
    true_pos = {x: i for i, x in enumerate(true_order)}

    total_pairs = 0
    correct_pairs = 0

    for a, b in combinations(predicted, 2):
        total_pairs += 1
        if (true_pos[a] < true_pos[b] and predicted.index(a) < predicted.index(b)) or \
           (true_pos[a] > true_pos[b] and predicted.index(a) > predicted.index(b)):
            correct_pairs += 1

    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

model_name = sys.argv[1]
match model_name:
    case "InternVL3-14B" | "InternVL3-8B":
        from base.InternVL import InternVL as ModelClass
    case "InternVL2_5-8B-MPO" | "InternVL2_5-4B-MPO":
        from base.InternVL2 import InternVL2 as ModelClass
    case "Ovis2-16B" | "Ovis2-8B" | "Ovis2-4B":
        from base.Ovis2 import Ovis2 as ModelClass
    case "Qwen2.5-VL-7B-Instruct":
        from base.Qwen2 import Qwen2 as ModelClass
    case "MiniCPM-o-2_6":
        from base.MiniCPM import MiniCPM as ModelClass
    case "Kimi-VL-A3B-Instruct":
        from base.Kimi_VL import Kimi_VL as ModelClass
    case x:
        print("-> Invalid model name:", x)
        exit()

print(f"-> Loading model: {model_name}")
model_path = os.path.join(MODEL_DIR, model_name)
model = ModelClass(model_path, PROMPT)
csvs = [os.path.join(CSV_DIR, name) for name in os.listdir(CSV_DIR)]

# Set seed for reproducibility
np.random.seed(42)


for csv in csvs:
    output_path = f"output/{model_name}_{os.path.basename(csv)}.json"
    print(output_path)
    if os.path.exists(output_path):
        continue
    outputs = []
    print(f"-> Processing {csv}")
    df = pd.read_csv(csv)[["T0_IMG","T1_IMG", "T2_IMG", "T3_IMG", "T4_IMG"]]
    for index, row in df.iterrows():
            perm = np.random.permutation(4)
            err = False
            for name in row:
                if not os.path.exists(os.path.join(IMG_DIR, name)):
                    err = True
                    break
            if err:
                continue
            _row = row.iloc[1:5]
            paths = [os.path.join(IMG_DIR, row.iloc[0])] + [os.path.join(IMG_DIR, name) for name in _row.iloc[perm].values]
            print(paths)
            output = model.order_images(paths)
            print(f"-> {index} Output: {output}")
            outputs.append([index, output, perm.tolist()])
    with open(output_path, "w") as json_file:
        json.dump(outputs, json_file)
    print('Output written to', output_path)
