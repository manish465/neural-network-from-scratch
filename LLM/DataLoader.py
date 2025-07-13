import torch
from torch.utils.data import Dataset, DataLoader
from Tokenizer import BPE


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer: BPE, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)
            # add BOS and EOS
            ids = [tokenizer.vocab["<bos>"]] + ids + [tokenizer.vocab["<eos>"]]
            # split into sequences
            for i in range(0, len(ids) - seq_len):
                input_ids = ids[i : i + seq_len]
                target_ids = ids[i + 1 : i + seq_len + 1]
                self.data.append(
                    (
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(target_ids, dtype=torch.long),
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(texts, batch_size=32, seq_len=128, shuffle=True, vocab_size=30000):
    tokenizer = BPE(vocab_size=vocab_size)
    tokenizer.train(texts)
    dataset = TextDataset(texts, tokenizer, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), tokenizer


# Example usage:
# texts = ["Hello world", "Another example sentence..."]
# loader, tok = get_dataloader(texts)
# for x, y in loader:
#     print(x.shape, y.shape)
