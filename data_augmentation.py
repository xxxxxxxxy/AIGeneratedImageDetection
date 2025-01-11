from data_processing import load_train_val_datasets
from model import MultiModalMultiTaskModel
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
import os
from PIL import Image


def predict_and_add_pseudo_labels(test_file, model, tokenizer, feature_extractor, train_dataset, val_dataset, confidence_threshold=0.9):
    """
    使用当前模型对 updated_test_data 进行预测，将高置信度的样本加入训练集，扩充数据分布。

    Args:
    - test_file (str): 测试数据文件路径。
    - model (torch.nn.Module): 已训练的模型。
    - tokenizer (AutoTokenizer): 分词器。
    - train_dataset (datasets.Dataset): 当前训练集。
    - confidence_threshold (float): 伪标签的置信度阈值。

    Returns:
    - extended_train_dataset: 扩充后的训练集。
    """
    
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
    

    test_df = pd.read_excel(test_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  
    
    
    def predict_with_confidence(caption, image_path):
        """
        Predict labels for a single data sample.
        
        Args:
            caption (str): Text description.
            image_path (str): Path to the image.
        
        Returns:
            tuple: Predicted Label_A and Label_B.
        """
        text_embeddings = tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        try:
            image = Image.open(image_path).convert("RGB")
            image_embeddings = feature_extractor(images=[image], return_tensors="pt")
        except Exception as e:
            print(f"Error loading image: {image_path}. Using placeholder tensor.")
            image_embeddings = {
                "pixel_values": torch.zeros((1, 3, 224, 224))  # Placeholder for missing/corrupt image
            }
      
        inputs = {
            "input_ids": text_embeddings["input_ids"].to(device),
            "attention_mask": text_embeddings["attention_mask"].to(device),
            "pixel_values": image_embeddings["pixel_values"].to(device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            
        logits_A = torch.sigmoid(outputs["logits_A"])
        logits_B = torch.softmax(outputs["logits_B"], dim=1)
        
        label_A_pred = logits_A.item() > 0.5  # 二分类，> 0.5 为 1
        confidence_A = logits_A.item() if label_A_pred else 1 - logits_A.item()

        label_B_pred = torch.argmax(logits_B, dim=1).item()
        confidence_B = logits_B[0, label_B_pred].item()

        return int(label_A_pred), confidence_A, label_B_pred, confidence_B
        

    # 保存高置信度伪标签样本
    pseudo_labels = []
    for index, row in test_df.iterrows():
        caption = row["Caption"]
        image_path = os.path.join('./Test_updated' ,'image_'+str(row['Index'])+'.jpg')
        label_A_pred, confidence_A, label_B_pred, confidence_B = predict_with_confidence(caption, image_path)
        
        # 如果两个标签均满足置信度要求，加入伪标签集
        if confidence_A >= confidence_threshold and confidence_B >= confidence_threshold:
            pseudo_labels.append({
                "index": index,
                "Caption": caption,
                "LABEL_A": label_A_pred,
                "LABEL_B": label_B_pred,
                "Image_path": image_path,
            })

    # 将伪标签样本加入训练集
    if pseudo_labels:
        pseudo_dataset = Dataset.from_list(pseudo_labels)
        print("开始map")
        pseudo_dataset = pseudo_dataset.map(preprocess_data, batched=True, batch_size=256)
            
        # pseudo_dataset.set_format("torch", columns=["input_ids", "attention_mask", "pixel_values", "LABEL_A", "LABEL_B"])
        
        train_test_split = pseudo_dataset.train_test_split(test_size=0.2, seed=42)

        pseudo_train_dataset = train_test_split['train']
        pseudo_val_dataset = train_test_split['test']
       
        pseudo_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "pixel_values", "LABEL_A", "LABEL_B"])
        pseudo_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "pixel_values", "LABEL_A", "LABEL_B"])
            
        # 合并
        extended_train_dataset = concatenate_datasets([train_dataset, pseudo_train_dataset])
        extended_val_dataset = concatenate_datasets([val_dataset, pseudo_val_dataset])
        
    else:
        print("No pseudo labels meet the confidence threshold.")
        extended_train_dataset = train_dataset
        extended_val_dataset = val_dataset

    return extended_train_dataset, extended_val_dataset


def train_augmentation(train_dataset, val_dataset, tokenizer, feature_extractor):
    test_file = './Test_updated/captions.xlsx'
    # best_model_dir = "./best_model"

    # 加载模型时，使用自定义模型类
    best_model = MultiModalMultiTaskModel("bert-base-uncased", "openai/clip-vit-base-patch32", num_labels_B=6)
    best_model.load_state_dict(torch.load("./best_model/pytorch_model.bin"))

    # train_dataset, val_dataset, tokenizer, feature_extractor = load_train_val_datasets()
    
    extended_train_dataset, extended_val_dataset = predict_and_add_pseudo_labels(test_file, best_model, tokenizer, feature_extractor, train_dataset, val_dataset, confidence_threshold=0.8)
    # extended_val_dataset= predict_and_add_pseudo_labels(test_file, best_model, tokenizer, feature_extractor, val_dataset, confidence_threshold=0.999)
    return extended_train_dataset, extended_val_dataset