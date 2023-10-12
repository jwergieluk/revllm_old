import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn.functional as F

class DistilBertByHand:
    
    def __init__(self,split_heads_first=True):
        
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_weights = self.model.state_dict()
        self.split_heads_first = split_heads_first #For experimentation with splitting heads first or second
        self.embedding_dimension = 768
        self.num_heads = 12
        self.head_size = self.embedding_dimension // self.num_heads
        self.tokens_len = None  # This will be initialized during embedding
        self.hidden_state_list = []
        self.hidden_states = None
        self.attention_weights_list = []
        self.attention_weights = None
        self.mine_for_comparison = None #just exists during my experimentation

    def layer_norm(self, x, weight, bias, eps=1e-12):
        
        mean = x.mean(dim=-1, keepdim=True)
        # variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True) #An alternate for comparison
        # std_dev = (variance + eps).sqrt()
        std_dev = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std_dev + eps)
        output = weight * x_normalized + bias
        
        return output

    def get_head_tensor_split_first(self, X, layer, Q_K_or_V):
        
        # For pytorch broadcasting to work, we need to expand the tensor to (1, self.token_length, 768)
        X_expanded = X.unsqueeze(0)
        
        #Weight matrix W_Q, W_K, or W_V
        weight_matrix = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.' + Q_K_or_V.lower() + '_lin.weight']
            
        head_divided_weight_matrix = weight_matrix.view(self.num_heads, self.head_size, self.embedding_dimension)
        if layer == 0:
            print("head_divided_weight_matrix shape: ", head_divided_weight_matrix.shape)

        #Bias matrix b_Q, b_K, or b_V
        bias_matrix = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.' + Q_K_or_V.lower() + '_lin.bias']

        head_divided_bias_matrix = bias_matrix.view(self.num_heads, self.head_size)
        
        # Multiply X with W_Q, W_K, or W_V
        head_matrices = torch.matmul(X_expanded, head_divided_weight_matrix.transpose(1, 2)) + head_divided_bias_matrix.unsqueeze(1)
            
        # Reshape to get the head tensor
        head_matrices = head_matrices.squeeze(1)
        
        return head_matrices
    
    def get_head_tensor_split_second(self, X, layer, Q_K_or_V):
        
        # Weight matrix W_Q, W_K, or W_V
        weight_matrix = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.' + Q_K_or_V.lower() + '_lin.weight']
        
        # Bias matrix b_Q, b_K, or b_V
        bias = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.' + Q_K_or_V.lower() + '_lin.bias']
        
        # Multiply X with W_Q, W_K, or W_V
        output_tensor = torch.matmul(X, weight_matrix) + bias
        
        # Reshape to split into multiple heads
        output_tensor = output_tensor.view(self.num_heads, self.tokens_len, self.head_size)
        
        return output_tensor


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
        
        self.hidden_state_list.append(X)
        
        return X

    def attention(self, X, layer):
    
        # Query, Key, and Value heads
        # Currently have it two ways for experimentation
        if self.split_heads_first:        
            Q = self.get_head_tensor_split_first(X, layer, 'Q')
            K = self.get_head_tensor_split_first(X, layer, 'K')
            V = self.get_head_tensor_split_first(X, layer, 'V')
        else:
            Q = self.get_head_tensor_split_second(X, layer, 'Q')
            K = self.get_head_tensor_split_second(X, layer, 'K')
            V = self.get_head_tensor_split_second(X, layer, 'V')        
        if layer == 0:
            print("Q shape: ", Q.shape)
            print("K shape: ", K.shape)
            print("V shape: ", V.shape)
            
        # Attention Weights
        A = torch.softmax(torch.matmul(Q, K.transpose(1, 2) / torch.sqrt(torch.tensor(self.head_size).float())), dim=-1)
        self.attention_weights_list.append(A)
        if layer == 0:
            print("A shape: ", A.shape)
        
        #--------------------------------start of divergence--------------------------------------------------

        # Update V
        V = torch.matmul(A, V)
        if layer == 0:
            print("V shape: ", V.shape)
            
        # Concatenating the heads
        V = V.view(self.tokens_len, self.embedding_dimension)
        
        # V = V.permute(0, 1, 2)
        # V = V.contiguous().view(self.tokens_len, self.embedding_dimension)

        
        if layer == 0:
            print("V shape: ", V.shape)  
            
        # Linear layer
        W_out_lin = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.out_lin.weight']
        # W_out_lin = W_out_lin.transpose(0,1) #TODO: Need this or as is?
        b_out_lin = self.distilbert_weights['transformer.layer.' + str(layer) + '.attention.out_lin.bias']
        
        residual = torch.matmul(V, W_out_lin) + b_out_lin
        
        #--------------------------------(hopefully) end of divergence-----------------------------------------
        
        # Residual Connections
        X = X + residual
        
        return X

    def feed_forward(self, X, layer):
        
        # FF Linear 1
        W_ff_l1 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin1.weight']
        b_ff_l1 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin1.bias']
        # b_ff_l1_matrix = b_ff_l1.repeat(self.tokens_len, 1) #todo: delete ?
        
        FF_data = torch.matmul(X, W_ff_l1.transpose(0, 1)) + b_ff_l1
        
        # FF ReLU
        FF_data = F.gelu(FF_data)
        
        # FF Linear 2
        W_ff_l2 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin2.weight']
        b_ff_l2 = self.distilbert_weights['transformer.layer.' + str(layer) + '.ffn.lin2.bias']
        
        X = torch.matmul(FF_data, W_ff_l2.transpose(0, 1)) + b_ff_l2
        
        # Normalize
        W_ff = self.distilbert_weights['transformer.layer.' + str(layer) + '.output_layer_norm.weight']
        b_ff = self.distilbert_weights['transformer.layer.' + str(layer) + '.output_layer_norm.bias']
                
        X = self.layer_norm(X, W_ff, b_ff)
        
        self.hidden_state_list.append(X)
        
        return X

    def run_layers(self, X):
        
        for layer in range(6):
            
            # Attention
            X = self.attention(X, layer)
            
            # Sa Normalize
            W_sa = self.distilbert_weights['transformer.layer.' + str(layer) + '.sa_layer_norm.weight']
            b_sa = self.distilbert_weights['transformer.layer.' + str(layer) + '.sa_layer_norm.bias']
            X = self.layer_norm(X, W_sa, b_sa)
            
            # Feed Forward
            X = self.feed_forward(X, layer)
        
        self.attention_weights = tuple(self.attention_weights_list)
            
        return X
    
    def __call__(self, sentence):
        
        X = self.embed(sentence)
        
        X = self.run_layers(X)
        
        self.hidden_states = tuple(self.hidden_state_list)
        
        return X
