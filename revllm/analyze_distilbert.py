#todo:
# Make a separate visualization module
# Common library adds (among many more): SHAP, LIME, LRP

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
from captum.attr import LayerIntegratedGradients#, LayerConductance

class qAndA():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  def __init__(self, model_name:str, preprocessor:object):

    self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    self.model.to(self.device)
    self.model.eval()
    self.model.zero_grad()
    self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    self.preprocessor = preprocessor
    self.start_scores = None
    self.end_scores = None
  
  def predict(self, input_ids=None, attention_mask = None, position:int = 0, show_prediction:bool = False) -> None:

    if input_ids is None:
        output = self.model(self.preprocessor.input_ids, attention_mask=self.preprocessor.attention_mask)
    else: #for use in lig.attribute below, which need these args to be passed in
        output = self.model(input_ids, attention_mask=attention_mask)

    self.start_scores = output.start_logits
    self.end_scores = output.end_logits

    if show_prediction == True:

        print('        Question: ', self.preprocessor.question)
        print('Predicted Answer: ', ' '.join(self.preprocessor.all_tokens[torch.argmax(self.start_scores) : torch.argmax(self.end_scores)+1]))
        print('   Actual Answer: ', self.preprocessor.ground_truth)
    
    pred = (self.start_scores, self.end_scores)
    pred = pred[position]

    return pred.max(1).values

  #---------------------------------------- Helper function for color map---------------------------------------------

  def _summarize_attributions_internal(self, attributions:torch.Tensor) -> torch.Tensor:

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions

  #---------------------------- ---------------------------------------------------------------------------------------

  def lig_color_map(self):

    lig = LayerIntegratedGradients(self.predict, self.model.distilbert.embeddings)

    attributions_start, delta_start = lig.attribute(inputs=self.preprocessor.input_ids,
                                                    baselines=self.preprocessor.baseline_input_ids,
                                                    additional_forward_args=(self.preprocessor.attention_mask,0,False),
                                                    return_convergence_delta=True)
    attributions_end, delta_end = lig.attribute(inputs=self.preprocessor.input_ids, 
                                                baselines=self.preprocessor.baseline_input_ids,
                                                additional_forward_args=(self.preprocessor.attention_mask,1,False),
                                                return_convergence_delta=True)

    attributions_start_sum = self._summarize_attributions_internal(attributions_start)
    attributions_end_sum = self._summarize_attributions_internal(attributions_end)

    # storing couple samples in an array for visualization purposes
    start_position_vis = viz.VisualizationDataRecord(
                            attributions_start_sum,
                            torch.max(torch.softmax(self.start_scores[0], dim=0)),
                            torch.argmax(self.start_scores),
                            torch.argmax(self.start_scores),
                            str(self.preprocessor.ground_truth_start_ind),
                            attributions_start_sum.sum(),       
                            self.preprocessor.all_tokens,
                            delta_start)

    end_position_vis = viz.VisualizationDataRecord(
                            attributions_end_sum,
                            torch.max(torch.softmax(self.end_scores[0], dim=0)),
                            torch.argmax(self.end_scores),
                            torch.argmax(self.end_scores),
                            str(self.preprocessor.ground_truth_end_ind),
                            attributions_end_sum.sum(),       
                            self.preprocessor.all_tokens,
                            delta_end)

    print('\033[1m', 'Visualizations For Start Position', '\033[0m')
    viz.visualize_text([start_position_vis])

    print('\033[1m', 'Visualizations For End Position', '\033[0m')
    viz.visualize_text([end_position_vis])

  #-------------------------------------- Helper function for top_k_tokens---------------------------------------------

  def _get_top_k_attributed_tokens_internal(self, attrs: torch.Tensor, k:int) -> Tuple[List[str], torch.Tensor, torch.Tensor]:

    values, indices = torch.topk(attrs, k)
    top_tokens = [self.preprocessor.all_tokens[idx] for idx in indices]

    return top_tokens, values, indices

  #---------------------------- ---------------------------------------------------------------------------------------

  def lig_top_k_tokens(self, k:int=5) -> None:

    #todo: this is all a repeat - need it again?
    lig2 = LayerIntegratedGradients(self.predict, [self.model.distilbert.embeddings.word_embeddings])

    attributions_start = lig2.attribute(inputs=(self.preprocessor.input_ids),
                                      baselines=(self.preprocessor.baseline_input_ids),
                                      additional_forward_args=(self.preprocessor.attention_mask,0,False))
    attributions_end = lig2.attribute(inputs=(self.preprocessor.input_ids),
                                      baselines=(self.preprocessor.baseline_input_ids),
                                      additional_forward_args=(self.preprocessor.attention_mask, 1, False))

    attributions_start_word = self._summarize_attributions_internal(attributions_start[0])
    attributions_end_word = self._summarize_attributions_internal(attributions_end[0])

    top_words_start, top_words_val_start, top_word_ind_start = self._get_top_k_attributed_tokens_internal(attributions_start_word,k=k)
    top_words_end, top_words_val_end, top_words_ind_end = self._get_top_k_attributed_tokens_internal(attributions_end_word,k=k)

    df_start = pd.DataFrame({'Word(Index), Attribution': ["{} ({}), {}".format(word, pos, round(val.item(),2)) for word, pos, val in zip(top_words_start, top_word_ind_start, top_words_val_start)]})
    df_start.style.set_properties(cell_ids=False)

    df_end = pd.DataFrame({'Word(Index), Attribution': ["{} ({}), {}".format(word, pos, round(val.item(),2)) for word, pos, val in zip(top_words_end, top_words_ind_end, top_words_val_end)]})
    df_end.style.set_properties(cell_ids=False)

    full_token_list = ['{}({})'.format(token, str(i)) for i, token in enumerate(self.preprocessor.all_tokens)]

    print(f"Full token list: {full_token_list}")
    print(f"Top 5 attributed embeddings for start position: {df_start}")
    print(f"Top 5 attributed embeddings for end position: {df_end}")