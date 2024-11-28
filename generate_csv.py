import pandas as pd
import cv2
import os
import re

def load_images_from_path(path, number):
    images = []
    srcs = []
    pattern = rf"^{number}(?!\d)"
    
    for filename in os.listdir(path):
        if re.match(pattern, filename):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                images.append(img)
                srcs.append(filename)

    return images, srcs


def load_images(number):
    final_images = []
    final_srcs = []

    final_path = f"data/final_submissions/{number}/"
    web_path = "data/web/"
    ai_path = "data/ai/"

    if not os.path.exists(final_path):
        print(f"The final submissions path '{final_path}' does not exist.")
        return None, None, None

    for filename in os.listdir(final_path):
        img = cv2.imread(os.path.join(final_path, filename))
        if img is not None:
            final_images.append(img)
            final_srcs.append(f"{number}_{filename}")

    web_images, web_srcs = load_images_from_path(web_path, number)
    ai_images, ai_srcs = load_images_from_path(ai_path, number)

    if not final_images:
        print(f"No images found in '{final_path}'. Please check the contents.")
        return None, None, None

    if not web_images and not ai_images:
        print(
            f"The number '{number}' does not correspond to any valid images in 'web' or 'ai' folders."
        )
        return None, None, None

    if not web_images:
        print(f"No web images found with prefix '{number}' in '{web_path}'.")
        return None, None, None
    if not ai_images:
        print(f"No AI images found with prefix '{number}' in '{ai_path}'.")
        return None, None, None

    return final_images, web_images, ai_images, final_srcs, web_srcs, ai_srcs

def read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
    

resnets_temp=[]
new_columns = ["Final_Submission", "Inspiration", "Similarity"]

for i in range(1,26):
    resnets_temp.append(read_csv(f"csv/csv2/{i}.csv"))
#Temp fix cause we still dont have all data for 26 group
resnets_temp.append(read_csv(f"csv/csv2/27.csv"))

resnets = pd.concat(resnets_temp, ignore_index=True)
resnets.columns = new_columns
resnets["Final_Submission"] = resnets["Final_Submission"].apply(lambda x: "_".join(x.split('/')[-2:]))
resnets["Inspiration"] = resnets["Inspiration"].apply(lambda x: x.split('/')[-1])


colors = read_csv("csv/color_similarity.csv")

dinos=read_csv("csv/dino_similarity.csv")
dinos.columns = new_columns
dinos["Final_Submission"] = dinos["Final_Submission"].apply(lambda x: "_".join(x.split('\\')[-2:]))
dinos["Inspiration"] = dinos["Inspiration"].apply(lambda x: x.split('\\')[-1])

contrasts = read_csv("csv/contrast_similarity.csv")
contrasts.columns = new_columns
contrasts["Final_Submission"] = contrasts["Final_Submission"].apply(lambda x: "_".join(x.split('\\')[-2:]))
contrasts["Inspiration"] = contrasts["Inspiration"].apply(lambda x: x.split('\\')[-1])


csv_files =[
    colors,
    resnets,
    dinos,
    contrasts
    ]
def zero_one_scale(x:pd.Series):
    return (x-x.min())/(x.max()-x.min())

def standard_scale(x:pd.Series):
    return (x-x.mean())/x.std()


for file in csv_files:
    file["Similarity"] = zero_one_scale(file["Similarity"])
    
metrics = ["Color Similarity", "ResNet Similarity", "Dino Similarity", "Contrast Similarity"]

# Combine each metric into a single DataFrame with a new 'Metric' column
dfs = []
for csv_file, metric in zip(csv_files, metrics):
    df = csv_file.copy()
    df['Metric'] = metric  # Add a column for metric type
    dfs.append(df)

# Concatenate all DataFrames
all_data = pd.concat(dfs, ignore_index=True)

merge_all = pd.merge(
    pd.merge(
        colors,
        resnets,
        on=["Final_Submission", "Inspiration"],
        suffixes=("_colors", "_resnets")),
    pd.merge(
        dinos,
        contrasts,
        on=["Final_Submission", "Inspiration"],
        suffixes=("_dinos", "_contrasts")),
    on=["Final_Submission", "Inspiration"],
)
merge_all.columns = ["Final_Submission", "Inspiration", "Color_Similarity", "ResNet_Similarity", "Dino_Similarity","Contrast_Similarity"]
merge_all.to_csv("csv/all_similarity.csv", index=False)

ai_srcs_all = []
web_srcs_all = []

for i in range(1, 26):
    _, _, _, _, web_srcs, ai_srcs = load_images(i)
    ai_srcs_all.extend(ai_srcs)
    web_srcs_all.extend(web_srcs)

_, _, _, _, web_srcs, ai_srcs = load_images(27)
ai_srcs_all.extend(ai_srcs)
web_srcs_all.extend(web_srcs)

web_df = merge_all[merge_all['Inspiration'].isin(web_srcs_all)].copy()
ai_df = merge_all[merge_all['Inspiration'].isin(ai_srcs_all)].copy()
    
web_df.loc[:, 'Source'] = 'Web'
ai_df.loc[:, 'Source'] = 'AI'
    
combined_df = pd.concat([web_df, ai_df])

all_final_src = []
for group in range(1 , 26):
    _, _, _, final_src, _, _ = load_images(group)
    all_final_src.extend(final_src)
all_final_src.extend(load_images(27)[3])

combined_df = combined_df[combined_df['Final_Submission'].isin(all_final_src)]
    
combined_df.to_csv("csv/all_similarities_with_srcs.csv", index=False)