from data_processing import load_train_val_datasets
from model import MultiModalMultiTaskModel
import torch
from evaluation import predict_and_save_multimodal
from data_augmentation import train_augmentation
import json
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from data_processing import load_train_val_datasets  
from model import MultiModalMultiTaskModel  
from evaluation import compute_metrics
from transformers import AutoConfig

# 加载数据集
print("开始加载数据集")
train_dataset, val_dataset, tokenizer, feature_extractor = load_train_val_datasets()  # Assume this function loads preprocessed datasets

# 数据扩充
print("开始扩充数据集")
train_dataset_aug, val_dataset_aug = train_augmentation(train_dataset, val_dataset, tokenizer, feature_extractor)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    # eval_steps=5000,  # 设置触发评估的步数
    learning_rate=2e-5,
    per_device_train_batch_size=256, 
    per_device_eval_batch_size=256,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir='./logs',
    label_names=["LABEL_A", "LABEL_B"],
    save_strategy="epoch",  # 保存策略为每隔指定步数
    # save_steps=5000,  # 每1000步保存一次模型
    save_total_limit=1,  # 只保存最近的3个检查点
    load_best_model_at_end=True,  # 在训练结束时加载最佳模型
    metric_for_best_model="f1_A",  # 用于选择最佳模型的指标
    greater_is_better=True  # 指标越高越好
)

# 初始化模型
num_labels_B = 6  # 5 + 1 
text_model_name = "bert-base-uncased"
image_model_name = "openai/clip-vit-base-patch32"
model = MultiModalMultiTaskModel(text_model_name, image_model_name, num_labels_B)

# # 初始化 Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(text_model_name)

# 自定义数据整理函数
def data_collator(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels_A = torch.tensor([f["LABEL_A"] for f in features])
    labels_B = torch.tensor([f["LABEL_B"] for f in features])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "LABEL_A": labels_A,
        "LABEL_B": labels_B
    }

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_aug,
    eval_dataset=val_dataset_aug,#_aug!!!!!!!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator  # 使用自定义的数据整理器
)

# 开始训练
trainer.train()

# 保存最佳模型
best_model_dir = "./best_model_aug"
trainer.save_model(best_model_dir)
torch.save(model.state_dict(), "./best_model_aug/pytorch_model.bin")


# 验证模型性能
results = trainer.evaluate()
print("Evaluation Results:", results)



predict_and_save_multimodal(
    test_file="./Test_updated/captions.xlsx",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
)

# 读取 JSON 文件
with open('answer.json', 'r') as file:
    data = json.load(file)

# 将 Label_A 为 0 的对应的 Label_B 也置为 0
for item in data:
    if item["Label_A"] == 0:
        item["Label_B"] = 0

# 将处理后的数据写回文件
with open('data_processed.json', 'w') as file:
    json.dump(data, file, indent=4)

print("处理完成，结果已保存到 data_processed.json")