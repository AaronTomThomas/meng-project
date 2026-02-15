import pytest
import torch
from models.transformer import TransformerEncoder


class TestTransformer:
    """Test suite for transformer model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_length = 64
        self.vocab_size = 96
        self.d_model = 128
        self.n_layers = 2
    
    def test_transformer_encoder_output_shape(self):
        """Test that TransformerEncoder produces correct output shape."""
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        model = TransformerEncoder(n_layers=self.n_layers)
        y = model(x)
        
        assert y.shape == (self.batch_size, self.seq_length, self.d_model)
    
    def test_transformer_encoder_with_attention_weights(self):
        """Test that TransformerEncoder returns correct shapes with attention weights."""
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        model = TransformerEncoder(n_layers=self.n_layers)
        y, weights = model(x, return_weights=True)
        
        assert y.shape == (self.batch_size, self.seq_length, self.d_model)
        assert weights.shape == (self.n_layers, self.batch_size, 4, self.seq_length, self.seq_length)