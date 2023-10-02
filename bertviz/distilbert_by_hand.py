import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn.functional as F

class DistilBertByHand:
    
    def __init__(self):
        
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_weights = self.model.state_dict()
        self.embedding_dimension = 768
        self.num_heads = 6
        self.head_size = self.embedding_dimension // self.num_heads
        self.tokens_len = None  # This will be initialized during embedding

    def layer_norm(self, x, weight, bias, eps=1e-12):
        
        mean = x.mean(dim=-1, keepdim=True)
        std_dev = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std_dev + eps)
        output = weight * x_normalized + bias
        
        return output

    def get_head_tensor(self, X_expanded, layer, Q_K_or_V):
        
        #Weight matrix W_Q, W_K, or W_V
        weight_matrix = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.' + Q_K_or_V.lower() + '_lin.weight']
        head_divided_weight_matrix = weight_matrix.view(self.num_heads, self.head_size, self.embedding_dimension)
                
        #Bias matrix b_Q, b_K, or b_V
        bias_matrix = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.' + Q_K_or_V.lower() + '_lin.bias']
        head_divided_bias_matrix = bias_matrix.view(self.num_heads, self.head_size)
        
        # Multiply X with W_Q, W_K, or W_V
        head_matrices = torch.matmul(X_expanded, head_divided_weight_matrix.transpose(1, 2)) + head_divided_bias_matrix.unsqueeze(1)
        
        # Reshape to get the head tensor
        head_matrices = head_matrices.squeeze(1)
        
        return head_matrices

    def embed(self, sentence):
        
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = inputs["input_ids"][0]
        
        # Initialize tokens_len
        self.tokens_len = len(inputs)
        
        # Full token embeddings
        W = self.distilbert_weights['embeddings.word_embeddings.weight']
        
        # Sentence token embeddings
        X = W[inputs]
        
        # Positional embeddings
        P_full = self.distilbert_weights['embeddings.position_embeddings.weight']
        P = P_full[:self.tokens_len, :]
        
        # Add position embeddings to token embeddings
        X = X + P
        
        # Normalize
        X = self.layer_norm(X, self.distilbert_weights['embeddings.LayerNorm.weight'], self.distilbert_weights['embeddings.LayerNorm.bias'])
        
        return X

    def attention(self, X, layer):
        
        # For pytorch broadcasting to work, we need to expand the tensor to (1, self.token_length, 768)
        X_expanded = X.unsqueeze(0)
        
        # Query, Key, and Value heads
        Q = self.get_head_tensor(X_expanded, layer, 'Q')
        K = self.get_head_tensor(X_expanded, layer, 'K')
        V = self.get_head_tensor(X_expanded, layer, 'V')
        
        # Attention Weights
        A = torch.softmax(torch.matmul(Q, K.transpose(1, 2) / torch.sqrt(torch.tensor(self.head_size).float())), dim=-1)
        
        # Update V
        V = torch.matmul(A, V)
        
        # Concatenating the heads
        V = V.view(self.tokens_len, self.embedding_dimension)
        
        # Linear layer
        W_out_lin = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.out_lin.weight']
        b_out_lin = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.out_lin.bias']
        b_out_lin_matrix = b_out_lin.repeat(self.tokens_len, 1)
        
        residual = torch.matmul(V, W_out_lin) + b_out_lin_matrix #TODO: Need W_out_lin.transpose(0,1) as per copilot suggestion?
        # residual = torch.matmul(V, W_out_lin.transpose(0,1)) + b_out_lin_matrix 

        # Residual Connections
        X = X + residual
        
        # Normalize
        W_sa = self.distilbert_weights['transformer.layer.' + str(layer) + '.sa_layer_norm.weight']
        b_sa = self.distilbert_weights['transformer.layer.' + str(layer) + '.sa_layer_norm.bias']
        
        X = self.layer_norm(X, W_sa, b_sa)
        
        return X

    def feed_forward(self, X, layer):
        
        # FF Linear 1
        W_ff_l1 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin1.weight']
        b_ff_l1 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin1.bias']
        b_ff_l1_matrix = b_ff_l1.repeat(self.tokens_len, 1)
        
        FF_data = torch.matmul(X, W_ff_l1.transpose(0, 1)) + b_ff_l1_matrix
        
        # FF ReLU
        FF_data = F.gelu(FF_data)
        
        # FF Linear 2
        W_ff_l2 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin2.weight']
        b_ff_l2 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin2.bias']
        b_ff_l2_matrix = b_ff_l2.repeat(self.tokens_len, 1)
        
        X = torch.matmul(FF_data, W_ff_l2.transpose(0, 1)) + b_ff_l2_matrix
        
        # Normalize
        W_ff = self.distilbert_weights['transformer.layer.' + str(layer) + '.output_layer_norm.weight']
        b_ff = self.distilbert_weights['transformer.layer.' + str(layer) + '.output_layer_norm.bias']
        
        X = self.layer_norm(X, W_ff, b_ff)
        
        return X

    def run_layers(self, X):
        
        for layer in range(6):
        
            X = self.attention(X, layer)
            X = self.feed_forward(X, layer)
            
        return X
    
    def __call__(self, sentence):
        
        X = self.embed(sentence)
        
        X = self.run_layers(X)
        
        return X