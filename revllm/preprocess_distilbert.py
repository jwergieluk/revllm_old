import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch

from typing import Optional, Tuple, List 

from transformers import (DistilBertTokenizer, 
                          DistilBertForQuestionAnswering
                          )

from captum.attr import visualization as viz
# from captum.attr import LayerIntegratedGradients, LayerConductance

class PreprocessQAndA():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def __init__(self, model_path:str):

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
    self.ground_truth_start_index = None
    self.ground_truth_end_index = None

  def __call__(self, question:str, context:str, ground_truth:str) -> Tuple[torch.Tensor, torch.Tensor, int]:

    #save as variables to call in analyzer
    self.question = question
    self.context = context
    self.ground_truth = ground_truth

    #tokenize inputs (question + context, with special tokens)
    question_ids = self.tokenizer.encode(question, add_special_tokens=False)
    context_ids = self.tokenizer.encode(context, add_special_tokens=False)        
    input_ids_raw_list = [self.cls_token_id] + question_ids + [self.sep_token_id] + context_ids + [self.sep_token_id]

    #tokenize baseline (independent of input - necessary for integrated gradients)
    baseline_input_ids_raw_list = [self.cls_token_id] + [self.baseline_token_id] * len(question_ids) + [self.sep_token_id] + \
        [self.baseline_token_id] * len(context_ids) + [self.sep_token_id]

    #convert input and baseline to tensors
    #input_ids need to be a (1,input_length) tensor for the model
    self.input_ids = torch.tensor([input_ids_raw_list], device=self.device)
    self.baseline_input_ids = torch.tensor([baseline_input_ids_raw_list], device=self.device)

    #create attention mask tensor from input_ids
    self.attention_mask = torch.ones_like(self.input_ids)

    #get all tokens
    token_indices = self.input_ids[0].detach().tolist()
    self.all_tokens = self.tokenizer.convert_ids_to_tokens(token_indices)

    #get ground truth tokens
    ground_truth_tokens = self.tokenizer.encode(self.ground_truth, add_special_tokens=False)

    #identify start and end indices of ground truth
    self.ground_truth_end_index = token_indices.index(ground_truth_tokens[-1])
    self.ground_truth_start_index = self.ground_truth_end_index - len(ground_truth_tokens) + 1

class PreprocessSentiment():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  labels = ['negative', 'positive']

  def __init__(self, model_path:str):

    self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    self.baseline_token_id = self.tokenizer.pad_token_id
    self.sep_token_id = self.tokenizer.sep_token_id
    self.cls_token_id = self.tokenizer.cls_token_id
    self.context = None
    self.input_ids = None
    self.ground_truth=None
    self.ground_truth_index=None
    self.baseline_input_ids = None
    self.attention_mask = None
    self.all_tokens = None


  def __call__(self, context:str,ground_truth:str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    
    self.context = context
    self.ground_truth = ground_truth
    self.ground_truth_index = self.labels.index(ground_truth)

    #tokenize inputs (question + context, with special tokens)
    context_ids = self.tokenizer.encode(context, add_special_tokens=False)
    input_ids_raw_list = [self.cls_token_id] + context_ids + [self.sep_token_id]
    #note: same as simply using the tokeinzer, but we keep it explicit 

    #tokenize baseline (independent of input - necessary for integrated gradients)
    baseline_input_ids_raw_list = [self.cls_token_id] + [self.baseline_token_id] * len(context_ids) + [self.sep_token_id]

    #convert input and baseline to tensors
    self.input_ids = torch.tensor([input_ids_raw_list], device=self.device)
    self.baseline_input_ids = torch.tensor([baseline_input_ids_raw_list], device=self.device)

    #create attention mask tensor from input_ids
    self.attention_mask = torch.ones_like(self.input_ids)

    #get all tokens
    token_indices = self.input_ids[0].detach().tolist()
    self.all_tokens = self.tokenizer.convert_ids_to_tokens(token_indices)

class PreprocessMaskedLM():
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    def __init__(self, model_path:str):
  
      self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
      self.baseline_token_id = self.tokenizer.pad_token_id
      self.sep_token_id = self.tokenizer.sep_token_id
      self.cls_token_id = self.tokenizer.cls_token_id
      self.unmasked_context = None
      self.context = None
      self.ground_truth=None
      self.ground_truth_index=None
      self.input_ids = None
      self.baseline_input_ids = None
      self.attention_mask = None
      self.all_tokens = None
      self.mask_index = None
  
    def __call__(self, unmasked_context:str) -> Tuple[torch.Tensor, torch.Tensor, int]:
      
      while True:

        masked_substring = input('Choose a token to mask: ')
        if masked_substring in unmasked_context:
            context = unmasked_context.replace(masked_substring, '[MASK]')
            print('         Context:', context)
            print('Unmasked context:', unmasked_context)
            break
        else:
            print('This substring is not in the context. Please choose another.')   

      self.context = context
      self.unmasked_context = unmasked_context
      self.ground_truth = masked_substring

      #tokenize inputs (question + context, with special tokens)
      context_ids = self.tokenizer.encode(self.context, add_special_tokens=False)
      input_ids_raw_list = [self.cls_token_id] + context_ids + [self.sep_token_id]
      self.mask_index = input_ids_raw_list.index(self.tokenizer.mask_token_id)
      self.ground_truth_index = self.mask_index #Name both for consistency in analyze_distilbert

      #tokenize baseline (independent of input - necessary for integrated gradients)
      baseline_input_ids_raw_list = [self.cls_token_id] + [self.baseline_token_id] * len(context_ids) + [self.sep_token_id]
  
      #convert input and baseline to tensors
      self.input_ids = torch.tensor([input_ids_raw_list], device=self.device)
      self.baseline_input_ids = torch.tensor([baseline_input_ids_raw_list], device=self.device)
  
      #create attention mask tensor from input_ids
      self.attention_mask = torch.ones_like(self.input_ids)
  
      #get all tokens
      token_indices = self.input_ids[0].detach().tolist()
      self.all_tokens = self.tokenizer.convert_ids_to_tokens(token_indices)