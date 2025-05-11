"""
Encoder-based model architecture for Hangman letter prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel, AutoModel, AutoConfig
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HangmanBERTConfig(BertConfig):
    """
    Configuration class for HangmanBERT.
    """
    vocab_size: int = 32  # a-z, '*', and special tokens
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 50
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    cls_token_id: int = 28
    sep_token_id: int = 27
    mask_token_id: int = 29
    wrong_attempts_size: int = 6  # 0-5 wrong attempts


class HangmanBERTModel(BertPreTrainedModel):
    """
    BERT-based model for Hangman letter prediction.
    """
    config_class = HangmanBERTConfig

    def __init__(self, config: HangmanBERTConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 26)  # 26 possible next letters
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token indices.
            attention_mask: Mask to avoid attending to padding tokens.
            token_type_ids: Segment token indices.
            position_ids: Indices of positions.
            return_dict: Whether to return a dictionary of outputs.
            
        Returns:
            Tuple of (logits, hidden_states, attentions).
        """
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).float()
        
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )
        
        # Use CLS token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classify next letter
        logits = self.classifier(pooled_output)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        if return_dict:
            return {
                'logits': logits,
                'probs': probs,
                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions
            }
        else:
            return logits, probs, outputs.hidden_states, outputs.attentions


class HangmanTransformerModel(nn.Module):
    """
    Transformer-based model for Hangman letter prediction using a custom architecture.
    Suitable for training from scratch without BERT pre-training.
    """
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 50,
    ):
        super().__init__()
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 26)  # 26 possible next letters
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for the model.
        """
        # Initialize embeddings
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token indices.
            attention_mask: Mask to avoid attending to padding tokens.
            
        Returns:
            Probability distribution over next letters.
        """
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()  # Assuming 0 is pad token
        
        # Create position IDs
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Apply TransformerEncoder
        # Convert attention_mask from [0, 1] to boolean mask where 1 = keep, 0 = mask
        src_key_padding_mask = (attention_mask == 0)
        
        # Pass through encoder
        encoder_output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling
        # Mask out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoder_output)
        masked_output = encoder_output * mask_expanded
        sum_embeddings = torch.sum(masked_output, dim=1)
        seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = sum_embeddings / seq_lengths
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classify next letter
        logits = self.classifier(pooled_output)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs


class HangmanCNN1DModel(nn.Module):
    """
    CNN-based model for Hangman letter prediction.
    Uses 1D convolutions for sequence processing.
    """
    def __init__(
        self,
        vocab_size: int = 32,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        # Fully connected layers
        total_filters = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(total_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 26)  # 26 possible next letters
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token indices.
            
        Returns:
            Probability distribution over next letters.
        """
        # Embed the input
        embedded = self.embeddings(input_ids)  # (batch, seq_len, embedding_dim)
        
        # Transpose for convolution
        embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # Apply convolutions
        conv_outputs = [self.relu(conv(embedded)) for conv in self.convs]
        
        # Apply global max pooling
        pooled_outputs = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in conv_outputs]
        
        # Concatenate pooled outputs
        concat = torch.cat(pooled_outputs, dim=1)
        
        # Apply dropout
        concat = self.dropout(concat)
        
        # Apply fully connected layers
        hidden = self.relu(self.fc1(concat))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs


class HangmanPretrainedBERTModel(nn.Module):
    """
    Hangman model that uses the prajjwal1/bert-small pre-trained model.
    Truncates the embedding layer to match the Hangman vocabulary size.
    """
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        
        # Load pre-trained model
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-small")
        
        # Truncate the embeddings to match our vocabulary size
        original_embeddings = self.bert.embeddings.word_embeddings
        self.bert.embeddings.word_embeddings = nn.Embedding(
            vocab_size, 
            original_embeddings.embedding_dim,
            padding_idx=original_embeddings.padding_idx
        )
        
        # Initialize truncated embeddings with values from pre-trained embeddings
        # This preserves the embedding values for tokens that exist in both vocabularies
        with torch.no_grad():
            # Only copy weights for the first vocab_size tokens
            # assuming token IDs are aligned (common with special tokens first)
            self.bert.embeddings.word_embeddings.weight[:vocab_size] = \
                original_embeddings.weight[:vocab_size]
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 26)  # 26 possible next letters
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token indices.
            attention_mask: Mask to avoid attending to padding tokens.
            token_type_ids: Segment token indices.
            position_ids: Indices of positions.
            return_dict: Whether to return a dictionary of outputs.
            
        Returns:
            Dictionary with logits, probabilities, hidden states, and attentions.
        """
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()  # Assuming 0 is pad token
        
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
        )
        
        # Use CLS token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Classify next letter
        logits = self.classifier(pooled_output)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        if return_dict:
            return {
                'logits': logits,
                'probs': probs,
                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions
            }
        else:
            return logits, probs, outputs.hidden_states, outputs.attentions


def create_model(model_type: str, config: Optional[dict] = None) -> nn.Module:
    """
    Create a model based on the specified type.
    
    Args:
        model_type: Type of model to create (bert, pretrained_bert, transformer, cnn).
        config: Configuration for the model.
        
    Returns:
        Model instance.
    """
    if model_type == 'bert':
        if config is None:
            config = {}
        bert_config = HangmanBERTConfig(**config)
        return HangmanBERTModel(bert_config)
    
    elif model_type == 'pretrained_bert':
        if config is None:
            config = {}
        return HangmanPretrainedBERTModel(**config)
    
    elif model_type == 'transformer':
        if config is None:
            config = {}
        return HangmanTransformerModel(**config)
    
    elif model_type == 'cnn':
        if config is None:
            config = {}
        return HangmanCNN1DModel(**config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")