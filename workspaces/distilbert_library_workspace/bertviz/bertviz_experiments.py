# from bertviz import model_view, head_view
# from transformers import (BertTokenizer, BertModel, 
#                           RobertaTokenizer, RobertaModel, 
#                           GPT2Tokenizer, GPT2Model,
#                           DistilBertTokenizer, DistilBertForQuestionAnswering
# )

# class Weights:
#     def __init__(self, model_type, model_version):
#         self.model_type = model_type
#         self.model_version = model_version
#         self.attention_weights = None
#         self.tokens = None

#         if model_type == 'bert':
#             self.tokenizer = BertTokenizer.from_pretrained(model_version)
#             self.model = BertModel.from_pretrained(model_version, output_attentions=True)
#         elif model_type == 'roberta':
#             self.tokenizer = RobertaTokenizer.from_pretrained(model_version)
#             self.model = RobertaModel.from_pretrained(model_version, output_attentions=True)
#         elif model_type == 'gpt2':
#             self.tokenizer = GPT2Tokenizer.from_pretrained(model_version)
#             self.model = GPT2Model.from_pretrained(model_version, output_attentions=True)
#         elif model_type == 'distilbert_squad':
#             self.tokenizer = DistilBertTokenizer.from_pretrained(model_version)
#             self.model = DistilBertForQuestionAnswering.from_pretrained(model_version, output_attentions=True)
#         else:
#             raise ValueError("Model type not recognized.")
    
#     def run_model(self, sentence):
#         inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
#         input_ids = inputs['input_ids']
        
#         if self.model_type == 'gpt2':
#             outputs = self.model(input_ids)
#             attention = outputs[-1]
#         else:  # For BERT and RoBERTa
#             outputs = self.model(**inputs)
#             attention = outputs.attentions
        
#         self.attention_weights = attention
#         self.tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
#         self.token_order = {token: idx for idx, token in enumerate(self.tokens)}
#         new_token_order = {}
#         for token, order in bert_weights.token_order.items():
#             new_key = token[1:] if token.startswith('Ġ') else token
#             new_token_order[new_key] = order
#         self.token_order = new_token_order    
        
#     def model_view_visualize(self, sentence):
#         if self.attention_weights is None or self.tokens is None:
#             self.run_model(sentence)
#         model_view(self.attention_weights, self.tokens)
    
#     def head_view_visualize(self, sentence):
#         if self.attention_weights is None or self.tokens is None:
#             self.run_model(sentence)
#         head_view(self.attention_weights, self.tokens)
    
#     def get_attention_weights(self, sentence):
#         if self.attention_weights is None:
#             self.run_model(sentence)
#         return self.attention_weights
        
#     def get_specific_weight(self, sentence, layer, head, first_token, second_token):
#         if self.attention_weights is None:
#             self.run_model(sentence)
#         first_token_idx = self.token_order[first_token]
#         second_token_idx = self.token_order[second_token]
#         return self.attention_weights[layer][0][head][first_token_idx][second_token_idx]

from bertviz import model_view, head_view
from transformers import (
    DistilBertTokenizer, DistilBertModel,
    BertTokenizer, BertModel, 
    RobertaTokenizer, RobertaModel, 
    GPT2Tokenizer, GPT2Model, 
    XLNetTokenizer, XLNetModel,
    DistilBertForQuestionAnswering
)

from bertviz import BertVizBertVizModel
from bertviz.neuron_view import show

class Weights:
    def __init__(self, model_type, model_version):
        self.model_type = model_type
        self.model_version = model_version
        self.attention_weights = None
        self.tokens = None

        #todo: do it a better way.  This was just to get started initially
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model_version)
            self.model = BertModel.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_version)
            self.model = RobertaModel.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_version)
            self.model = GPT2Model.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_version)
            self.model = DistilBertModel.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained(model_version)
            self.model = XLNetModel.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'distilbert_squad': #todo: check on this
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_version)
            self.model = DistilBertForQuestionAnswering.from_pretrained(model_version, output_attentions=True)
        else:
            raise ValueError("Model type not recognized.")
    
    def run_model(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']

        if self.model_type in ['bert', 'roberta', 'distilbert','distilbert_squad']:
            outputs = self.model(**inputs)
            attention = outputs.attentions
        elif self.model_type == 'gpt2':
            outputs = self.model(input_ids)
            attention = outputs[-1]
        elif self.model_type == 'xlnet':
            # For XLNet, we might need to handle the permutation-based training and other specifics
            outputs = self.model(input_ids)
            attention = outputs.attentions
        else:
            raise ValueError("Model type not recognized.")
        
        self.attention_weights = attention
        self.tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
        self.token_order = {token: idx for idx, token in enumerate(self.tokens)}
        new_token_order = {}
        for token, order in bert_weights.token_order.items():
            new_key = token[1:] if token.startswith('Ġ') else token
            new_token_order[new_key] = order
        self.token_order = new_token_order    
        
    def model_view_visualize(self, sentence):
        if self.attention_weights is None or self.tokens is None:
            self.run_model(sentence)
        model_view(self.attention_weights, self.tokens)
    
    def head_view_visualize(self, sentence):
        if self.attention_weights is None or self.tokens is None:
            self.run_model(sentence)
        head_view(self.attention_weights, self.tokens)
        
    # since we're only using distilbert
    def neuron_view_visualize(self, sentence):
        if self.model_type != 'bert':
            print("Neuron view is currently supported only for BERT.")

        else:
            if self.attention_weights is None or self.tokens is None:
                self.run_model(sentence)
        
            # Use the BertViz-specific model and tokenizer for neuron view visualization
            neuron_model = BertVizModel.from_pretrained(self.model_version, output_attentions=True)
            neuron_tokenizer = BertVizTokenizer.from_pretrained(self.model_version)
        
            show(neuron_model, self.model_type, neuron_tokenizer, sentence)

    
    def get_attention_weights(self, sentence):
        if self.attention_weights is None:
            self.run_model(sentence)
        return self.attention_weights
        
    def get_specific_weight(self, sentence, layer, head, first_token, second_token):
        if self.attention_weights is None:
            self.run_model(sentence)
        first_token_idx = self.token_order[first_token]
        second_token_idx = self.token_order[second_token]
        return self.attention_weights[layer][0][head][first_token_idx][second_token_idx]

#------------------------------------------------------------------------------------------

# # For reference
# models_dict = {
#     'bert': 'bert-base-uncased',
#     'roberta': 'roberta-base',
#     'gpt2': 'gpt2-medium'
# }

# For reference
models_dict = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'gpt2': 'gpt2-medium',
    'xlnet': 'xlnet-base-cased'
}

# sentence = "The cat sat on the mat."
# model = 'distilbert'
# bert_weights = Weights(model, models_dict[model])
# bert_weights.head_view_visualize(sentence)
# bert_weights.model_view_visualize(sentence)
attention_data = bert_weights.get_attention_weights(sentence)

sentence = "The cat sat on the mat."
model = 'bert'
bert_weights = Weights(model, models_dict[model])
bert_weights.model_view_visualize(sentence)
bert_weights.head_view_visualize(sentence)
bert_weights.neuron_view_visualize(sentence)
bert_weights_ = bert_weights.get_attention_weights(sentence)

# Access and print all the model's weights (parameters)
# model_parameters = bert_weights.model.state_dict()

# for name, param in model_parameters.items():
#     print(name, param.size())

#------------------------------------------------------------------------------------------

print(f"model_type: {model}")
print(f"Number of layers: {len(attention_data)}")
print(f"Number of heads: {list(attention_data[0].shape)[1]}")

# print(bert_weights.get_specific_weight(sentence, 6, 3, 'The', 'cat').item())
# print(bert_weights.get_specific_weight(sentence, 6, 3, 'cat', 'The').item())


#------------------------------------------------------------------------------------------
# prep to add to visualize_distilbert.py
#------------------------------------------------------------------------------------------

from bertviz import model_view, head_view
from bertviz.neuron_view import show

#notes to self
model_type = 'distilbert'
model_version = 'distilbert-base-uncased'

class BertViz:
    def __init__(self, model_path: str, preprocessor):
        self.model_type = 'distilbert' #todo: do it a better way.  This was just to get started initially
        self.model_path = model_path
        self.prerprocessor = preprocessor
        self.model = DistilBertModel.from_pretrained(model_path, output_attentions=True)
        self.attention_weights = None
        self.tokens = None

        # self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    # def predict(self, input_ids=None, attention_mask = None, position:int = 0) -> None:

    #     output = self.model(self.preprocessor.input_ids, attention_mask=self.preprocessor.attention_mask)

    #     self.start_scores = output.start_logits
    #     self.end_scores = output.end_logits
             

    def run_model(self):

        outputs = self.model(self.preprocessor.input_ids, attention_mask=self.preprocessor.attention_mask)

        attention = outputs.attentions
        
        self.attention_weights = attention
        self.tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
   
        
    def model_view_visualize(self):
        if self.attention_weights is None or self.tokens is None:
            self.run_model()
        model_view(self.attention_weights, self.tokens)
    
    def head_view_visualize(self):
        if self.attention_weights is None or self.tokens is None:
            self.run_model()
        head_view(self.attention_weights, self.tokens)
        
# if bert later, then include neuron_view_visualize

    def get_attention_weights(self):
        if self.attention_weights is None:
            self.run_model()
        return self.attention_weights