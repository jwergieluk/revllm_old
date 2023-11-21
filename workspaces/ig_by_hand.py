import torch
import torch.nn as nn
import torch.optim as optim

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

input = "The movie had breathtaking visuals, but the storyline left a lot to be desired."

possible_outputs = ["negative", "positive"]

baseline_input_id = tokenizer.pad_token_id
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id

context_ids = tokenizer.encode(input, add_special_tokens=False)
input_ids_raw = [cls_token_id] + context_ids + [sep_token_id]
input_ids = torch.tensor([input_ids_raw])
attention_mask = torch.ones_like(input_ids)

baseline_input_ids_raw = [cls_token_id] + [baseline_input_id] * len(context_ids) + [sep_token_id]
baseline_input_ids = torch.tensor([baseline_input_ids_raw])

output = model(input_ids, attention_mask=attention_mask)
logits = output.logits
prediction_index = torch.argmax(logits,dim=1)
prediction = possible_outputs[prediction_index]
print(prediction)

def integrated_gradients_embedding_layer(model, baseline_embed, input_embed, target, steps=50):
    # Ensure model is in evaluation mode
    model.eval()

    # Generate interpolated embeddings between baseline and input embeddings
    interpolated_embeddings = [baseline_embed + (step / steps) * (input_embed - baseline_embed) 
                               for step in range(steps + 1)]

    # Compute gradients for each interpolated embedding
    gradients = []
    for interpolated_embed in interpolated_embeddings:
        interpolated_embed.requires_grad_(True)
        
        # Feed the interpolated embedding through the rest of the model
        output = model.forward_from_embedding_layer(interpolated_embed,attention_mask=attention_mask)

        # Zero out gradients from previous step
        model.zero_grad()

        # Backward pass
        output[0, target].backward()
        
        # Collect gradient
        gradients.append(interpolated_embed.grad.detach().clone())

    # Aggregate gradients and scale
    total_gradients = torch.sum(torch.stack(gradients), dim=0)
    scaled_integrated_gradients = (input_embed - baseline_embed) * total_gradients / (steps + 1)

    return scaled_integrated_gradients
