import re
from collections import Counter, defaultdict


class BPE:
    def __init__(self, vocab_size=30000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = {}
        self.bpe_codes = {}
        self.byte_encoder = {i: chr(i) for i in range(256)}
        self.byte_decoder = {chr(i): i for i in range(256)}
        self.token_pattern = re.compile(r"\S+")

    def get_stats(self, corpus):
        pairs = Counter()

        for word, freq in corpus.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        return pairs

    def merge_vocab(self, pair, corpus):
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_corpus = {}

        for word, freq in corpus.items():
            new_word = pattern.sub("".join(pair), word)
            new_corpus[new_word] = freq

        return new_corpus

    def train(self, texts):
        # Tokenize and count
        corpus = Counter()
        for text in texts:
            tokens = text.strip().split()
            for token in tokens:
                # add end-of-word marker
                word = " ".join(list(token)) + " </w>"
                corpus[word] += 1
        # initialize vocab
        vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        # learn BPE merges
        for i in range(self.vocab_size - len(vocab)):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.bpe_codes[best] = i
        # build final vocab
        for word in corpus:
            for token in word.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab = vocab
        self.inverse_vocab = {i: tok for tok, i in vocab.items()}

    def encode(self, text):
        tokens = []
        for word in text.strip().split():
            chars = list(word) + ["</w>"]
            while True:
                pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
                merge_candidates = {
                    pair: self.bpe_codes.get(pair, float("inf")) for pair in pairs
                }
                if not merge_candidates:
                    break
                best = min(merge_candidates, key=merge_candidates.get)
                if best not in self.bpe_codes:
                    break
                i = chars.index(best[0])
                chars[i : i + 2] = ["".join(best)]
            for tok in chars:
                tokens.append(self.vocab.get(tok, self.vocab["<unk>"]))
        return tokens

    def decode(self, token_ids):
        tokens = [self.inverse_vocab.get(i, "<unk>") for i in token_ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
