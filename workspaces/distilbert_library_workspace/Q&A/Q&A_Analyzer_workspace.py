#todo:
# Make a separate visualization module
# Common library adds (among many more): SHAP, LIME, LRP
# Automate preprocessing (if possible)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch

from typing import Optional, Tuple, List 

from transformers import (
    DistilBertForQuestionAnswering
    , DistilBertTokenizer
)

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients, LayerConductance

class DistilBertQandAAnalyzer():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def __init__(self, model_path:str):

    self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    self.model.to(self.device)
    self.model.eval()
    self.model.zero_grad()
    self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    self.ref_token_id = self.tokenizer.pad_token_id # A token used for generating token reference
    self.sep_token_id = self.tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    self.cls_token_id = self.tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    self.input_ids = None 
    self.ref_input_ids = None
    self.attention_mask = None
    self.all_tokens = None
    self.start_scores = None
    self.ground_truth_start_ind = None
    self.end_scores = None
    self.ground_truth_end_ind = None
    self.layer_attrs_start = None
    self.layer_attrs_end = None
    self.layer_attrs_start_dist = None
    self.layer_attrs_end_dist = None
    self.layer_attributions_start = None
    self.layer_attributions_end = None

  #--------------------------------------------------Helper Functions------------------------------------------------------

  def _predict(self, input_ids:Optional[torch.Tensor]=None, attention_mask:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:

    output = self.model(input_ids, attention_mask=attention_mask)

    return output.start_logits, output.end_logits

  def _squad_pos_forward_func(self, input_ids:Optional[torch.Tensor]=None, attention_mask:Optional[torch.Tensor]=None, position:int = 0) -> torch.Tensor:
    
    pred = self._predict(input_ids,attention_mask=attention_mask)
    pred = pred[position]
    
    return pred.max(1).values
  
  def _construct_input_ref_pair(self, question:str, text:str, ref_token_id:int, sep_token_id:int, cls_token_id:int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    
    question_ids = self.tokenizer.encode(question, add_special_tokens=False)
    text_ids = self.tokenizer.encode(text, add_special_tokens=False)
    local_input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]
    local_ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([local_input_ids], device=self.device), torch.tensor([local_ref_input_ids], device=self.device)#, len(question_ids)
    
  def _construct_attention_mask(self, local_input_ids:Optional[torch.Tensor]=None) -> torch.Tensor:

    return torch.ones_like(local_input_ids)
  
  def _construct_whole_bert_embeddings(self, local_input_ids, local_ref_input_ids) -> Tuple[torch.Tensor, torch.Tensor]:

    input_embeddings = self.model.distilbert.embeddings(local_input_ids)
    ref_input_embeddings = self.model.distilbert.embeddings(local_ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

  def _summarize_attributions(self, attributions:torch.Tensor) -> torch.Tensor:
      
      attributions = attributions.sum(dim=-1).squeeze(0)
      attributions = attributions / torch.norm(attributions)
      
      return attributions
  
  def _get_topk_attributed_tokens(self, attrs, k=5) -> Tuple[List[str], torch.Tensor, torch.Tensor]:

    values, indices = torch.topk(attrs, k)
    top_tokens = [self.all_tokens[idx] for idx in indices]

    return top_tokens, values, indices
  
  def _squad_pos_forward_func2(self, input_emb, attention_mask=None, position=0) -> torch.Tensor:
    pred = self.model(inputs_embeds=input_emb, attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values
  
  def _pdf_attr(self,attrs, bins=100) -> np.ndarray:
    return np.histogram(attrs, bins=bins, density=True)[0]
  
  def _prep_for_visualization(self, token_to_explain=None):
    self.layer_attrs_start = []
    self.layer_attrs_end = []

    self.layer_attrs_start_dist = []
    self.layer_attrs_end_dist = []

    input_embeddings, ref_input_embeddings = self._construct_whole_bert_embeddings(self.input_ids, self.ref_input_ids)

    for i in range(self.model.config.num_hidden_layers):
        lc = LayerConductance(self._squad_pos_forward_func2, self.model.distilbert.transformer.layer[i])
        self.layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(self.attention_mask, 0))
        self.layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(self.attention_mask, 1))
        self.layer_attrs_start.append(self._summarize_attributions(self.layer_attributions_start).cpu().detach().tolist())
        self.layer_attrs_end.append(self._summarize_attributions(self.layer_attributions_end).cpu().detach().tolist())
        
        if token_to_explain is not None:
            self.layer_attrs_start_dist.append(self.layer_attributions_start[0,token_to_explain,:].cpu().detach().tolist())
            self.layer_attrs_end_dist.append(self.layer_attributions_end[0,token_to_explain,:].cpu().detach().tolist())           

#-------------------------------------------------Wrapper Functions-----------------------------------------------------------------

  def __call__(self, question:str, text:str, ground_truth:str) -> None: #todo, from previous: , visualize:bool=True
    self.input_ids, self.ref_input_ids = self._construct_input_ref_pair(question, text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
    #todo, from previous: sep_id
    self.attention_mask = self._construct_attention_mask(self.input_ids)

    indices = self.input_ids[0].detach().tolist()
    self.all_tokens = self.tokenizer.convert_ids_to_tokens(indices)

    ground_truth_tokens = self.tokenizer.encode(ground_truth, add_special_tokens=False)
    self.ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
    self.ground_truth_start_ind = self.ground_truth_end_ind - len(ground_truth_tokens) + 1
  
    self.start_scores, self.end_scores = self._predict(self.input_ids, \
                                    attention_mask=self.attention_mask)

    print('        Question: ', question)
    print('Predicted Answer: ', ' '.join(self.all_tokens[torch.argmax(self.start_scores) : torch.argmax(self.end_scores)+1]))
    print('   Actual Answer: ', ground_truth)

#-------------------------------------------------------------------------------------------------------------------------------------

  def start_end_color_map(self) -> None:
    lig = LayerIntegratedGradients(self._squad_pos_forward_func, self.model.distilbert.embeddings)

    attributions_start, delta_start = lig.attribute(inputs=self.input_ids,
                                      baselines=self.ref_input_ids,
                                      additional_forward_args=(self.attention_mask, 0),
                                      return_convergence_delta=True)
    attributions_end, delta_end = lig.attribute(inputs=self.input_ids, baselines=self.ref_input_ids,
                                    additional_forward_args=(self.attention_mask, 1),
                                    return_convergence_delta=True)

    attributions_start_sum = self._summarize_attributions(attributions_start)
    attributions_end_sum = self._summarize_attributions(attributions_end)

    # storing couple samples in an array for visualization purposes
    start_position_vis = viz.VisualizationDataRecord(
                            attributions_start_sum,
                            torch.max(torch.softmax(self.start_scores[0], dim=0)),
                            torch.argmax(self.start_scores),
                            torch.argmax(self.start_scores),
                            str(self.ground_truth_start_ind),
                            attributions_start_sum.sum(),       
                            self.all_tokens,
                            delta_start)

    end_position_vis = viz.VisualizationDataRecord(
                            attributions_end_sum,
                            torch.max(torch.softmax(self.end_scores[0], dim=0)),
                            torch.argmax(self.end_scores),
                            torch.argmax(self.end_scores),
                            str(self.ground_truth_end_ind),
                            attributions_end_sum.sum(),       
                            self.all_tokens,
                            delta_end)

    print('\033[1m', 'Visualizations For Start Position', '\033[0m')
    viz.visualize_text([start_position_vis])

    print('\033[1m', 'Visualizations For End Position', '\033[0m')
    viz.visualize_text([end_position_vis])


  def top_5_tokens(self) -> None:

    lig2 = LayerIntegratedGradients(self._squad_pos_forward_func, \
                                    [self.model.distilbert.embeddings.word_embeddings])

    attributions_start = lig2.attribute(inputs=(self.input_ids),
                                      baselines=(self.ref_input_ids),
                                      additional_forward_args=(self.attention_mask, 0))
    attributions_end = lig2.attribute(inputs=(self.input_ids),
                                      baselines=(self.ref_input_ids),
                                      additional_forward_args=(self.attention_mask, 1))

    attributions_start_word = self._summarize_attributions(attributions_start[0])
    attributions_end_word = self._summarize_attributions(attributions_end[0])

    top_words_start, top_words_val_start, top_word_ind_start = self._get_topk_attributed_tokens(attributions_start_word)
    top_words_end, top_words_val_end, top_words_ind_end = self._get_topk_attributed_tokens(attributions_end_word)

    df_start = pd.DataFrame({'Word(Index), Attribution': ["{} ({}), {}".format(word, pos, round(val.item(),2)) for word, pos, val in zip(top_words_start, top_word_ind_start, top_words_val_start)]})
    df_start.style.set_properties(cell_ids=False)

    df_end = pd.DataFrame({'Word(Index), Attribution': ["{} ({}), {}".format(word, pos, round(val.item(),2)) for word, pos, val in zip(top_words_end, top_words_ind_end, top_words_val_end)]})
    df_end.style.set_properties(cell_ids=False)

    print(['{}({})'.format(token, str(i)) for i, token in enumerate(self.all_tokens)])
    print(f"Top 5 attributed embeddings for start position: {df_start}")
    print(f"Top 5 attributed embeddings for end position: {df_end}")

#-----------------------------------------------Visualizations-------------------------------------------------------------------------

  def visualize_layers(self) -> None:

    # todo: subdivide

    self._prep_for_visualization()

    fig, ax = plt.subplots(figsize=(15,5))
    xticklabels=self.all_tokens
    yticklabels=list(range(1,len(self.layer_attrs_start)+1))
    ax = sns.heatmap(np.array(self.layer_attrs_start), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Tokens')
    plt.ylabel('Layers')
    plt.show()

    fig, ax = plt.subplots(figsize=(15,5))
    xticklabels=self.all_tokens
    yticklabels=list(range(1,len(self.layer_attrs_start)+1))
    ax = sns.heatmap(np.array(self.layer_attrs_end), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2) #, annot=True
    plt.xlabel('Tokens')
    plt.ylabel('Layers')
    plt.show()


  def visualize_token(self,token_to_explain:int) -> None: #todo: make into a text entry

    self._prep_for_visualization(token_to_explain=token_to_explain) #todo: make so i don't have to call it twice

    fig, ax = plt.subplots(figsize=(20,10))
    ax = sns.boxplot(data=self.layer_attrs_start_dist)
    plt.xlabel('Layers')
    plt.ylabel('Attribution')
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))
    ax = sns.boxplot(data=self.layer_attrs_end_dist)
    plt.xlabel('Layers')
    plt.ylabel('Attribution')
    plt.show()
    
    layer_attrs_end_pdf = map(lambda single_attr:self._pdf_attr(single_attr), self.layer_attrs_end_dist)
    layer_attrs_end_pdf = np.array(list(layer_attrs_end_pdf))


    attr_sum = np.array(self.layer_attrs_end_dist).sum(-1)

    # size: #layers
    layer_attrs_end_pdf_norm = np.linalg.norm(layer_attrs_end_pdf, axis=-1, ord=1)

    #size: #bins x #layers
    layer_attrs_end_pdf = np.transpose(layer_attrs_end_pdf)

    #size: #bins x #layers
    layer_attrs_end_pdf = np.divide(layer_attrs_end_pdf, layer_attrs_end_pdf_norm, where=layer_attrs_end_pdf_norm!=0)

    fig, ax = plt.subplots(figsize=(20,10))
    plt.plot(layer_attrs_end_pdf)
    plt.xlabel('Bins')
    plt.ylabel('Density')
    plt.legend(['Layer '+ str(i) for i in range(1,len(self.layer_attrs_start)+1)])
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))

    # replacing 0s with 1s. np.log(1) = 0 and np.log(0) = -inf
    layer_attrs_end_pdf[layer_attrs_end_pdf == 0] = 1
    layer_attrs_end_pdf_log = np.log2(layer_attrs_end_pdf)

    # size: #layers
    entropies= -(layer_attrs_end_pdf * layer_attrs_end_pdf_log).sum(0)

    plt.scatter(np.arange(len(self.layer_attrs_start)), attr_sum, s=entropies * 100)
    plt.xlabel('Layers')
    plt.ylabel('Total Attribution')
    plt.show()

#-------------------------------------------cutting and pasting for now, will make work Thursday--------------------------------------------

# from bertviz import model_view, head_view
# from transformers import (
#     DistilBertTokenizer, DistilBertModel,
#     BertTokenizer, BertModel, 
#     RobertaTokenizer, RobertaModel, 
#     GPT2Tokenizer, GPT2Model, 
#     XLNetTokenizer, XLNetModel,
#     DistilBertForQuestionAnswering
# )
# from bertviz.neuron_view import show

# class Weights:
#   def __init__(self, model_type, model_version):
#     self.model_type = model_type
#     self.model_version = model_version
#     self.attention_weights = None
#     self.tokens = None

#     #todo: do it a better way.  This was just to get started initially
#     if model_type == 'bert':
#         self.tokenizer = BertTokenizer.from_pretrained(model_version)
#         self.model = BertModel.from_pretrained(model_version, output_attentions=True)
#     elif model_type == 'roberta':
#         self.tokenizer = RobertaTokenizer.from_pretrained(model_version)
#         self.model = RobertaModel.from_pretrained(model_version, output_attentions=True)
#     elif model_type == 'gpt2':
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_version)
#         self.model = GPT2Model.from_pretrained(model_version, output_attentions=True)
#     elif model_type == 'distilbert':
#         self.tokenizer = DistilBertTokenizer.from_pretrained(model_version)
#         self.model = DistilBertModel.from_pretrained(model_version, output_attentions=True)
#     elif model_type == 'xlnet':
#         self.tokenizer = XLNetTokenizer.from_pretrained(model_version)
#         self.model = XLNetModel.from_pretrained(model_version, output_attentions=True)
#     elif model_type == 'distilbert_squad': #todo: check on this
#         self.tokenizer = DistilBertTokenizer.from_pretrained(model_version)
#         self.model = DistilBertForQuestionAnswering.from_pretrained(model_version, output_attentions=True)
#     else:
#         raise ValueError("Model type not recognized.")
  
#   def run_model(self, sentence):
#     inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
#     input_ids = inputs['input_ids']

#     if self.model_type in ['bert', 'roberta', 'distilbert','distilbert_squad']:
#         outputs = self.model(**inputs)
#         attention = outputs.attentions
#     elif self.model_type == 'gpt2':
#         outputs = self.model(input_ids)
#         attention = outputs[-1]
#     elif self.model_type == 'xlnet':
#         # For XLNet, we might need to handle the permutation-based training and other specifics
#         outputs = self.model(input_ids)
#         attention = outputs.attentions
#     else:
#         raise ValueError("Model type not recognized.")
    
#     self.attention_weights = attention
#     self.tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

#     self.token_order = {token: idx for idx, token in enumerate(self.tokens)}
#     new_token_order = {}
#     for token, order in bert_weights.token_order.items():
#         new_key = token[1:] if token.startswith('Ġ') else token
#         new_token_order[new_key] = order
#     self.token_order = new_token_order    
      
#   def model_view_visualize(self, sentence):
#     if self.attention_weights is None or self.tokens is None:
#         self.run_model(sentence)
#     model_view(self.attention_weights, self.tokens)
  
#   def head_view_visualize(self, sentence):
#     if self.attention_weights is None or self.tokens is None:
#         self.run_model(sentence)
#     head_view(self.attention_weights, self.tokens)
      
#   def neuron_view_visualize(self, sentence):
#     if self.model_type != 'bert':
#       print("Neuron view is currently supported only for BERT.")

#     else:
#       if self.attention_weights is None or self.tokens is None:
#           self.run_model(sentence)
  
#       # Use the BertViz-specific model and tokenizer for neuron view visualization
#       neuron_model = BertVizModel.from_pretrained(self.model_version, output_attentions=True)
#       neuron_tokenizer = BertVizTokenizer.from_pretrained(self.model_version)
  
#       show(neuron_model, self.model_type, neuron_tokenizer, sentence)

  
#   def get_attention_weights(self, sentence):
#     if self.attention_weights is None:
#       self.run_model(sentence)
#     return self.attention_weights
      
#   def get_specific_weight(self, sentence, layer, head, first_token, second_token):
#     if self.attention_weights is None:
#       self.run_model(sentence)
#     first_token_idx = self.token_order[first_token]
#     second_token_idx = self.token_order[second_token]
#     return self.attention_weights[layer][0][head][first_token_idx][second_token_idx]
