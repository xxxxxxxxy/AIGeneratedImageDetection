import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoFeatureExtractor
import os
from PIL import Image
import torch


def load_train_val_datasets():
    # 加载训练数据
    train = pd.read_excel('./Train/captions.xlsx')

    train_dataset = []

    # 标签映射
    #['coco_image', 'sd3_image', 'sd21_image', 'sdxl_image', 'dalle_image', 'midjourney_image']
    model_name_to_label = {
        "sd21_image": 1,
        "sdxl_image": 2,
        "sd3_image": 3,
        "dalle_image": 4,
        "midjourney_image": 5
    }

    # 遍历数据并创建训练样本
    for i in range(len(train)):
        row = train.iloc[i]
        caption = row['Caption']
        real_image_path = os.path.join('./Train/coco_image' ,'image_'+str(row['Index'])+'.jpg')
        train_dataset.append({
            "Caption": f"Caption: {caption}",
            "Image_path": real_image_path,
            "LABEL_A": 0,
            "LABEL_B": 0  # 0表示'real_image'
        })

        for model_name, label_B_value in model_name_to_label.items():
            ai_image_path = './Train/' + str(model_name) +'/image_'+str(row['Index'])+'.jpg'
            train_dataset.append({
                "Caption": f"Caption: {caption}",
                "Image_path": ai_image_path,
                "LABEL_A": 1,
                "LABEL_B": model_name_to_label[model_name]
            })

    train_dataset = Dataset.from_list(train_dataset)#!!!!!!!!!!!!!!!!!!!!!!!!!![:10]
    
    
    
    #加载验证数据集
    val = pd.read_csv('./Validation/val_shuffle.csv')
    
    val_dataset = []
     # 遍历数据并创建训练样本
    for i in range(len(val)):
        row = val.iloc[i]
        caption = row['Caption']
        ai_image_path = os.path.join('./Validation' ,'image_'+str(row['Index'])+'.jpg')
        val_dataset.append({
            "Caption": f"Caption: {caption}",
            "Image_path": ai_image_path,
            "LABEL_A": row['Label_A'],
            "LABEL_B": row['Label_B']
        })
        
    val_dataset = Dataset.from_list(val_dataset)#!!!!!!!!!!!!!!!!!!!!!!!!!![:2]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    
    def preprocess_data(batch):
        text_embeddings = tokenizer(
            batch['Caption'],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        batch['input_ids'] = text_embeddings['input_ids']
        batch['attention_mask'] = text_embeddings['attention_mask']

        images = []
        for path in batch['Image_path']:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
            except:                
                images.append(torch.zeros((3, 224, 224)))
        image_embeddings = feature_extractor(images=images, return_tensors="pt")
        batch['pixel_values'] = image_embeddings['pixel_values']

        return batch

    train_dataset = train_dataset.map(preprocess_data, batched=True, batch_size=32)
    val_dataset = val_dataset.map(preprocess_data, batched=True, batch_size=32)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "pixel_values", "LABEL_A", "LABEL_B"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "pixel_values", "LABEL_A", "LABEL_B"])
    
    
    return train_dataset, val_dataset, tokenizer, feature_extractor