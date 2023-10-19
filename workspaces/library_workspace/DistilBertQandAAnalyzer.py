import numpy as np

import torch

from transformers import (
    DistilBertForQuestionAnswering
    , DistilBertTokenizer
)

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients

class DistilBertQandAAnalyzer():
  
  def __init__(self, model_path):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    self.model.to(self.device)
    self.model.eval()
    self.model.zero_grad()
    self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    self.ref_token_id = self.tokenizer.pad_token_id # A token used for generating token reference
    self.sep_token_id = self.tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    self.cls_token_id = self.tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

  def _predict(self, inputs, attention_mask=None):
    output = self.model(inputs, attention_mask=attention_mask)
    return output.start_logits, output.end_logits

  def _squad_pos_forward_func(self, inputs, attention_mask=None, position=0):
    pred = self._predict(inputs,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values
  
  def _construct_input_ref_pair(self, question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = self.tokenizer.encode(question, add_special_tokens=False)
    text_ids = self.tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device), len(question_ids)
    
  def _construct_attention_mask(self, input_ids):
      return torch.ones_like(input_ids)
  
  # Don't think I need this method any more
  # def _construct_whole_bert_embeddings(self, input_ids, ref_input_ids):
  #     input_embeddings = self.model.distilbert.embeddings(input_ids)
  #     ref_input_embeddings = self.model.distilbert.embeddings(ref_input_ids)
      
  #     return input_embeddings, ref_input_embeddings

  def _summarize_attributions(self, attributions):
      attributions = attributions.sum(dim=-1).squeeze(0)
      attributions = attributions / torch.norm(attributions)
      return attributions

  def __call__(self, question, text, ground_truth, visualize=True):

    input_ids, ref_input_ids, sep_id = self._construct_input_ref_pair(question, text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
    attention_mask = self._construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = self.tokenizer.convert_ids_to_tokens(indices)

    ground_truth_tokens = self.tokenizer.encode(ground_truth, add_special_tokens=False)
    ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
    ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1
    
    start_scores, end_scores = self._predict(input_ids, \
                                    attention_mask=attention_mask)

    print('Question: ', question)
    print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    lig = LayerIntegratedGradients(self._squad_pos_forward_func, self.model.distilbert.embeddings)

    attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      additional_forward_args=(attention_mask, 0),
                                      return_convergence_delta=True)
    attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                    additional_forward_args=(attention_mask, 1),
                                    return_convergence_delta=True)

    attributions_start_sum = self._summarize_attributions(attributions_start)
    attributions_end_sum = self._summarize_attributions(attributions_end)

    if visualize:
      # storing couple samples in an array for visualization purposes
      start_position_vis = viz.VisualizationDataRecord(
                              attributions_start_sum,
                              torch.max(torch.softmax(start_scores[0], dim=0)),
                              torch.argmax(start_scores),
                              torch.argmax(start_scores),
                              str(ground_truth_start_ind),
                              attributions_start_sum.sum(),       
                              all_tokens,
                              delta_start)

      end_position_vis = viz.VisualizationDataRecord(
                              attributions_end_sum,
                              torch.max(torch.softmax(end_scores[0], dim=0)),
                              torch.argmax(end_scores),
                              torch.argmax(end_scores),
                              str(ground_truth_end_ind),
                              attributions_end_sum.sum(),       
                              all_tokens,
                              delta_end)

      print('\033[1m', 'Visualizations For Start Position', '\033[0m')
      viz.visualize_text([start_position_vis])

      print('\033[1m', 'Visualizations For End Position', '\033[0m')
      viz.visualize_text([end_position_vis])
