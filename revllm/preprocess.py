import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch

from typing import Optional, Tuple, List 

from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients, LayerConductance

class DistilBertQandAPreprocessor():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def __init__(self, model_path:str):

    self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    self.model.to(self.device)
    self.model.eval()
    self.model.zero_grad()
    self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    self.baseline_token_id = self.tokenizer.pad_token_id
    self.sep_token_id = self.tokenizer.sep_token_id
    self.cls_token_id = self.tokenizer.cls_token_id
    self.question = None
    self.context = None
    self.ground_truth = None
    self.input_ids = None
    self.baseline_input_ids = None
    self.attention_mask = None
    self.all_tokens = None
    self.ground_truth_start_ind = None
    self.ground_truth_end_ind = None
    # self.layer_attrs_start = []
    # self.layer_attrs_end = []
    # self.layer_attrs_start_dist = []
    # self.layer_attrs_end_dist = []
    # self.layer_attributions_start = None
    # self.layer_attributions_end = None
    # self.input_embeddings = None
    # self.baseline_input_embeddings = None


  def __call__(self, question:str, context:str, ground_truth:str) -> Tuple[torch.Tensor, torch.Tensor, int]:

    #save as variables to call in analyzer
    self.question = question
    self.context = context
    self.ground_truth = ground_truth

    #tokenize inputs (question + context, with special tokens)
    question_ids = self.tokenizer.encode(question, add_special_tokens=False)
    context_ids = self.tokenizer.encode(context, add_special_tokens=False)        
    input_ids_raw = [self.cls_token_id] + question_ids + [self.sep_token_id] + context_ids + [self.sep_token_id]

    #tokenize baseline (independent of input - necessary for integrated gradients)
    baseline_input_ids_raw = [self.cls_token_id] + [self.baseline_token_id] * len(question_ids) + [self.sep_token_id] + \
        [self.baseline_token_id] * len(context_ids) + [self.sep_token_id]

    #convert input and baseline to tensors
    self.input_ids = torch.tensor([input_ids_raw], device=self.device)
    self.baseline_input_ids = torch.tensor([baseline_input_ids_raw], device=self.device)

    #----------------------------todo: understand this part better----------------------------------------

    #create attention mask tensor from input_ids
    self.attention_mask = torch.ones_like(self.input_ids)

    #get all tokens
    token_indices = self.input_ids[0].detach().tolist()
    self.all_tokens = self.tokenizer.convert_ids_to_tokens(token_indices)

#----------------------------------------------------------------------------------------------------------

    #get ground truth tokens
    ground_truth_tokens = self.tokenizer.encode(self.ground_truth, add_special_tokens=False)

    #identify start and end indices of ground truth
    self.ground_truth_end_ind = token_indices.index(ground_truth_tokens[-1])
    self.ground_truth_start_ind = self.ground_truth_end_ind - len(ground_truth_tokens) + 1