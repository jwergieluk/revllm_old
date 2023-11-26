import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st

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

st.write("Context:")
st.write(context_ids)
st.write("input_ids:")
st.write(input_ids)
st.write(input_ids.shape)
st.write("baseline_input_ids:")
st.write(baseline_input_ids)
st.write(baseline_input_ids.shape)
st.write("attention_mask:")
st.write(attention_mask)
st.write(attention_mask.shape)

with torch.no_grad():
    word_embeddings = model.distilbert.embeddings(input_ids)
    baseline_embeddings = model.distilbert.embeddings(baseline_input_ids)

st.write("word_embeddings shape:")
st.write(word_embeddings[0][0].mean())
st.write(word_embeddings[0][0].std())

seq_length = word_embeddings.size(1)
print(seq_length)
position_ids = torch.arange(seq_length, dtype=torch.long, device=word_embeddings.device).unsqueeze(0)
positional_embeddings = model.distilbert.embeddings.position_embeddings(position_ids)

# output = model(input_ids, attention_mask=attention_mask)
# logits = output.logits
# prediction_index = torch.argmax(logits,dim=1)
# prediction = possible_outputs[prediction_index]y
# print(prediction)
# print(input_embeddings)

def forward_from_embedding_layer(word_embeddings, positional_embeddings, attention_mask=None):
        # Sum word embeddings with positional embeddings
        # The positional embeddings can be accessed via `self.embeddings.position_embeddings`
        # Ensure the sequence lengths of embeddings and positional embeddings match

        x = word_embeddings + positional_embeddings

        # Process the sum through the transformer layers of DistilBERT
        for layer in transformer.layer:
            layer_outputs = layer(x, attention_mask=attention_mask)
            x = layer_outputs[0]  # The first element is the hidden state

        # Further processing as required
        return x

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
        output = forward_from_embedding_layer(interpolated_embed,attention_mask=attention_mask)

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

# ig = integrated_gradients_embedding_layer(model, baseline_embeddings, input_embeddings, prediction_index)
# print(ig)

# def forward_from_embedding_layer(self, embedding_output, attention_mask=None):
#     if attention_mask is None:
#         attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output.device)
#     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#     extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
#     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#     embedding_output = self.embeddings.LayerNorm(embedding_output)
#     embedding_output = self.embeddings.dropout(embedding_output)
#     sequence_output = self.transformer(embedding_output, extended_attention_mask)
#     return sequence_output[0]
