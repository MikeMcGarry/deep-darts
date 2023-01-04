import pandas as pd
import os

path = './dataset/annotations/'
label_files = os.listdir(path)

labels = []

for label_file in label_files:
    df = pd.read_pickle(os.path.join(path, label_file))
    df["img_folder"] = label_file.split(".")[0]
    labels.append(df)

labels = pd.concat(labels)

pd.to_pickle(obj=labels, filepath_or_buffer="dataset/labels.pkl")
