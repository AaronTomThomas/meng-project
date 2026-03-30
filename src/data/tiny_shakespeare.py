import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Char-level language modeling dataset.
    Produces (x, y) where y is x shifted by 1 (next-token prediction).
    """


    def __init__(self, text_path: str, block_size: int):
        self.block_size = block_size
        
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # encoder/decoder maps
        self.stoi = {ch : i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data

        self.n = len(self.data) - (self.block_size + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx : int):
        chunk = self.data[idx: idx + self.block_size + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
    

    def encode_text(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)
    
    def decode_tokens(self, t: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in t)
