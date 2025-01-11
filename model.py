import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel

class MultiModalMultiTaskModel(nn.Module):
    def __init__(self, text_model_name, image_model_name, num_labels_B):
        super(MultiModalMultiTaskModel, self).__init__()
        
        # Text Encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name, local_files_only=True)
        
        # Image Encoder
        self.image_encoder = CLIPModel.from_pretrained(image_model_name, local_files_only=True).vision_model
        
        # Combined 
        combined_hidden_size = self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size
        self.projection = nn.Linear(combined_hidden_size, 512)

        # Label_A Classifier
        self.classifier_A = nn.Linear(512, 1)
        
        # Label_B Classifier
        self.classifier_B = nn.Linear(512, num_labels_B)

    def forward(self, input_ids, attention_mask, pixel_values, LABEL_A=None, LABEL_B=None):

        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  # Use [CLS] token features
               
        image_output = self.image_encoder(pixel_values=pixel_values)
        image_features = image_output[1]  # Use the pooled output from the vision transformer
      
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Project 
        features = F.relu(self.projection(combined_features))
 
        logits_A = self.classifier_A(features)

        logits_B = self.classifier_B(features)

        loss = None

        if LABEL_A is not None and LABEL_B is not None:         
            loss_A = nn.BCEWithLogitsLoss()(logits_A, LABEL_A.unsqueeze(1).float())
                        
            loss_B_mask = (LABEL_A == 1).float()  
            if torch.sum(loss_B_mask) > 0:                
                loss_B = nn.CrossEntropyLoss(reduction='none')(logits_B, LABEL_B)
                loss_B = torch.sum(loss_B * loss_B_mask) / torch.sum(loss_B_mask) 
            else:
                loss_B = 0.0
            
            loss = loss_A + loss_B

        return {"loss": loss, "logits_A": logits_A, "logits_B": logits_B}


