from bertviz import model_view, head_view
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2Model

class Weights:
    def __init__(self, model_type, model_version):
        self.model_type = model_type
        self.model_version = model_version
        self.attention_weights = None
        self.tokens = None

        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model_version)
            self.model = BertModel.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_version)
            self.model = RobertaModel.from_pretrained(model_version, output_attentions=True)
        elif model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_version)
            self.model = GPT2Model.from_pretrained(model_version, output_attentions=True)
        else:
            raise ValueError("Model type not recognized.")
    
    def run_model(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        
        if self.model_type == 'gpt2':
            outputs = self.model(input_ids)
            attention = outputs[-1]
        else:  # For BERT and RoBERTa
            outputs = self.model(**inputs)
            attention = outputs.attentions
        
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

# For reference
models_dict = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'gpt2': 'gpt2-medium'
}

sentence = "The cat sat on the mat."
model = 'roberta'
bert_weights = Weights(model, models_dict[model])
bert_weights.head_view_visualize(sentence)
bert_weights.model_view_visualize(sentence)
attention_data = bert_weights.get_attention_weights(sentence)

#------------------------------------------------------------------------------------------

print(f"model_type: {model}")
print(f"Number of layers: {len(attention_data)}")
print(f"Number of heads: {list(attention_data[0].shape)[1]}")

print(bert_weights.get_specific_weight(sentence, 6, 3, 'The', 'cat').item())
print(bert_weights.get_specific_weight(sentence, 6, 3, 'cat', 'The').item())