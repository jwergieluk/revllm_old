#todo: clean

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

from captum.attr import LayerConductance

class DistilBertQandAVisualizer():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, model_path:str, preprocessor):

        self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.preprocessor = preprocessor
        self.layer_attrs_start = []
        self.layer_attrs_end = []
        self.layer_attrs_start_dist = []
        self.layer_attrs_end_dist = []
        self.layer_attributions_start = None
        self.layer_attributions_end = None

#-------------------------------------helper functions to visualize attributions-------------------------------------
    def _predict_for_visualization(self, input_embs:torch.Tensor, attention_mask:torch.Tensor, position:int = 0) -> torch.Tensor:
        pred = self.model(inputs_embeds=input_embs, attention_mask=attention_mask)
        pred = pred[position]
        return pred.max(1).values
    
    #repeat of analyze function
    def _summarize_attributions_internal(self, attributions:torch.Tensor) -> torch.Tensor:

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        return attributions
    
    def _prep_for_visualization(self, token_to_explain_index:int=None):
        
        #todo: this will not be necessary after separating out "token_to_explain_index" loop below
        self.layer_attrs_start = []
        self.layer_attrs_end = []
        self.layer_attrs_start_dist = []
        self.layer_attrs_end_dist = []
        self.layer_attributions_start = None
        self.layer_attributions_end = None
    
        input_embeddings = self.model.distilbert.embeddings(self.preprocessor.input_ids)
        baseline_input_embeddings = self.model.distilbert.embeddings(self.preprocessor.baseline_input_ids)

        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(self._predict_for_visualization, self.model.distilbert.transformer.layer[i])
            self.layer_attributions_start = lc.attribute(inputs=input_embeddings, 
                                                         baselines=baseline_input_embeddings, 
                                                         additional_forward_args=(self.preprocessor.attention_mask, 0)
                                                         )
            self.layer_attributions_end = lc.attribute(inputs=input_embeddings, 
                                                       baselines=baseline_input_embeddings, 
                                                       additional_forward_args=(self.preprocessor.attention_mask, 1)
                                                       )
            self.layer_attrs_start.append(self._summarize_attributions_internal(self.layer_attributions_start).cpu().detach().tolist())
            self.layer_attrs_end.append(self._summarize_attributions_internal(self.layer_attributions_end).cpu().detach().tolist())
            
            #todo: separate this out so I don't need to run _prep_for_visualization multiple times below
            if token_to_explain_index is not None:
                self.layer_attrs_start_dist.append(self.layer_attributions_start[0,token_to_explain_index,:].cpu().detach().tolist())
                self.layer_attrs_end_dist.append(self.layer_attributions_end[0,token_to_explain_index,:].cpu().detach().tolist())

#-------------------------------------------------------------------------------------------------------------------

    def lc_visualize_layers(self):
        # todo: subdivide

        self._prep_for_visualization()

        fig, ax = plt.subplots(figsize=(15,5))
        xticklabels=self.preprocessor.all_tokens
        yticklabels=list(range(1,len(self.layer_attrs_start)+1))
        ax = sns.heatmap(np.array(self.layer_attrs_start), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.title('Token attribution scores for start of answer')
        plt.show()

        fig, ax = plt.subplots(figsize=(15,5))
        xticklabels=self.preprocessor.all_tokens
        yticklabels=list(range(1,len(self.layer_attrs_start)+1))
        ax = sns.heatmap(np.array(self.layer_attrs_end), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2) #, annot=True
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.title('Token attribution scores for end of answer')
        plt.show()

#-------------------------------------helper functions for visualize token-------------------------------------

    def _pdf_attr(self,attrs, bins:int=100) -> np.ndarray:
        return np.histogram(attrs, bins=bins, density=True)[0]

    def _select_index(self,element):
    
        indices = [index for index, value in enumerate(self.preprocessor.all_tokens) if value == element]
        
        if not indices:
            print(f"{element} not found in the list.")
            return None
        
        if len(indices) == 1:
            return indices[0]
        
        while True:
            print(f"The element {element} occurs at indices: {indices}")
            chosen_index = int(input(f"Please select an index from the list above: "))
            if chosen_index in indices:
                return chosen_index
            else:
                print(f"Invalid choice. Please choose an index from {indices}")

#-------------------------------------------------------------------------------------------------------------
#TODO FIRST: SEE IF WORKS
    def lc_visualize_token(self,token_to_explain:str): 
        #todo: add option to visualize multiple tokens at once
        #todo: take str as input for token to explain
        #todo: subdivide

        token_to_explain_index = self._select_index(token_to_explain)
        
        self._prep_for_visualization(token_to_explain_index=token_to_explain_index) #todo: make so i don't have to call it twice

        fig, ax = plt.subplots(figsize=(20,10))
        ax = sns.boxplot(data=self.layer_attrs_start_dist)
        plt.title(f"Attribution scores of {token_to_explain} for start of answer")
        plt.xlabel('Layers')
        plt.ylabel('Attribution')
        plt.show()

        #----------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(20,10))
        ax = sns.boxplot(data=self.layer_attrs_end_dist)
        plt.title(f"Attribution scores of {token_to_explain} for end of answer")
        plt.xlabel('Layers')
        plt.ylabel('Attribution')
        plt.show()

        #----------------------------------------------------------------------------------------------

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

        #----------------------------------------------------------------------------------------------

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

