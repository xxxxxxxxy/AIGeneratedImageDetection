import json
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from transformers import AutoTokenizer, AutoFeatureExtractor
import numpy as np
import os
from PIL import Image


def compute_metrics(eval_pred):
    """
    自定义评估函数
    """
    logits, labels = eval_pred
    # print(labels)
    logits_A, logits_B = logits  
    labels_A, labels_B = labels  

    predictions_A = (torch.sigmoid(torch.tensor(logits_A)) > 0.5).int()
    predictions_B = np.argmax(logits_B, axis=1)

    metrics = {}

    # label_A
    accuracy_A = accuracy_score(labels_A, predictions_A)
    f1_A = f1_score(labels_A, predictions_A, average="binary")
    precision_A = precision_score(labels_A, predictions_A, average="binary")
    recall_A = recall_score(labels_A, predictions_A, average="binary")
    
    metrics["accuracy_A"] = accuracy_A
    metrics["f1_A"] = f1_A
    metrics["precision_A"] = precision_A
    metrics["recall_A"] = recall_A

    # label_B
    accuracy_B = accuracy_score(labels_B, predictions_B)
    f1_B = f1_score(labels_B, predictions_B, average="weighted")  # 对于多分类，使用 weighted 平均
    precision_B = precision_score(labels_B, predictions_B, average="weighted")
    recall_B = recall_score(labels_B, predictions_B, average="weighted")
    
    metrics["accuracy_B"] = accuracy_B
    metrics["f1_B"] = f1_B
    metrics["precision_B"] = precision_B
    metrics["recall_B"] = recall_B

    return metrics



def predict_and_save_multimodal(test_file, model, tokenizer, feature_extractor, output_file="answer.json"):
    """
    Predict labels for multi-modal data and save results to a JSON file.
    
    Args:
        test_file (str): Path to the test file (CSV with 'Caption' and 'Image_path' columns).
        model (torch.nn.Module): The multi-modal multi-task model.
        tokenizer (AutoTokenizer): Tokenizer for text inputs.
        feature_extractor (AutoFeatureExtractor): Feature extractor for image inputs.
        output_file (str): Path to save the prediction results in JSON format.
    """
    test_df = pd.read_excel(test_file)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!![:2]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  
    model.eval()

    def predict(caption, image_path):
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
                "pixel_values": torch.zeros((1, 3, 224, 224))  # 图像读不出来的占位符
            }

        # Move inputs to the correct device
        inputs = {
            "input_ids": text_embeddings["input_ids"].to(device),
            "attention_mask": text_embeddings["attention_mask"].to(device),
            "pixel_values": image_embeddings["pixel_values"].to(device)
        }

        # Predict using the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Process predictions
        label_A_pred = torch.sigmoid(outputs["logits_A"]).item() > 0.5  
        label_B_pred = torch.argmax(outputs["logits_B"], dim=1).item()  
        
        return int(label_A_pred), label_B_pred

    print("开始预测......")
    results = []
    for index, row in test_df.iterrows():
        # print(index)
        caption = row["Caption"]
        image_path = os.path.join('./Test_updated' ,'image_'+str(row['Index'])+'.jpg')
        label_A_pred, label_B_pred = predict(caption, image_path)
        results.append({
            "index": index,
            "Caption": caption,
            # "Image_path": image_path,
            "Label_A": label_A_pred,
            "Label_B": label_B_pred
        })

    print("写入中......")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {output_file}")

