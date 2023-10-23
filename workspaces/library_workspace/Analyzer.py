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

  # Helper Functions

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

    return torch.tensor([local_input_ids], device=self.device), torch.tensor([local_ref_input_ids], device=self.device), len(question_ids)
    
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

# Wrapper Functions

  def __call__(self, question:str, text:str, ground_truth:str, visualize:bool=True) -> None:
    self.input_ids, self.ref_input_ids, sep_id = self._construct_input_ref_pair(question, text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
    
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


  def visualize_start_end(self) -> None:
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


  def top_5(self) -> None:

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


  def visualize_layers(self,token_to_explain:int=23) -> None:
    layer_attrs_start = []
    layer_attrs_end = []

    # The token that we would like to examine separately.
    layer_attrs_start_dist = []
    layer_attrs_end_dist = []

    input_embeddings, ref_input_embeddings = self._construct_whole_bert_embeddings(self.input_ids, self.ref_input_ids)

    for i in range(self.model.config.num_hidden_layers):
        lc = LayerConductance(self._squad_pos_forward_func2, self.model.distilbert.transformer.layer[i])
        layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(self.attention_mask, 0))
        layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(self.attention_mask, 1))
        layer_attrs_start.append(self._summarize_attributions(layer_attributions_start).cpu().detach().tolist())
        layer_attrs_end.append(self._summarize_attributions(layer_attributions_end).cpu().detach().tolist())

        # storing attributions of the token id that we would like to examine in more detail in token_to_explain
        layer_attrs_start_dist.append(layer_attributions_start[0,token_to_explain,:].cpu().detach().tolist())
        layer_attrs_end_dist.append(layer_attributions_end[0,token_to_explain,:].cpu().detach().tolist())
    
    fig, ax = plt.subplots(figsize=(15,5))
    xticklabels=self.all_tokens
    yticklabels=list(range(1,len(layer_attrs_start)+1))
    ax = sns.heatmap(np.array(layer_attrs_start), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Tokens')
    plt.ylabel('Layers')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(15,5))
    xticklabels=self.all_tokens
    yticklabels=list(range(1,len(layer_attrs_start)+1))
    ax = sns.heatmap(np.array(layer_attrs_end), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2) #, annot=True
    plt.xlabel('Tokens')
    plt.ylabel('Layers')
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))
    ax = sns.boxplot(data=layer_attrs_start_dist)
    plt.xlabel('Layers')
    plt.ylabel('Attribution')
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))
    ax = sns.boxplot(data=layer_attrs_end_dist)
    plt.xlabel('Layers')
    plt.ylabel('Attribution')
    plt.show()
    
    layer_attrs_end_pdf = map(lambda layer_attrs_end_dist:self._pdf_attr(layer_attrs_end_dist), layer_attrs_end_dist)
    layer_attrs_end_pdf = np.array(list(layer_attrs_end_pdf))


    attr_sum = np.array(layer_attrs_end_dist).sum(-1)

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
    plt.legend(['Layer '+ str(i) for i in range(1,len(layer_attrs_start)+1)])
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))

    # replacing 0s with 1s. np.log(1) = 0 and np.log(0) = -inf
    layer_attrs_end_pdf[layer_attrs_end_pdf == 0] = 1
    layer_attrs_end_pdf_log = np.log2(layer_attrs_end_pdf)

    # size: #layers
    entropies= -(layer_attrs_end_pdf * layer_attrs_end_pdf_log).sum(0)

    plt.scatter(np.arange(len(layer_attrs_start)), attr_sum, s=entropies * 100)
    plt.xlabel('Layers')
    plt.ylabel('Total Attribution')
    plt.show()