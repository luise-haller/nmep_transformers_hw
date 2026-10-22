from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""

        # for encoding: char -> idx
        for ch in self.characters:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        # for decoding: idx -> char
        self.inverse_vocab = {i: ch for ch, i in self.vocab.items()}

        if verbose:
            print("Vocabulary:", self.vocab)

        # raise NotImplementedError("Need to implement vocab initialization")

    def encode(self, text: str) -> torch.Tensor:
        # raise NotImplementedError(
        #     "Need to implement encoder that converts text to tensor of tokens."
        # )
        text = text.lower()
        tokens = [self.vocab[ch] for ch in text]
        return torch.tensor(tokens, dtype=torch.long)


    def decode(self, tokens: torch.Tensor) -> str:
        # raise NotImplementedError(
        #     "Need to implement decoder that converts tensor of tokens to text."
        # )
        return "".join(self.inverse_vocab[i] for i in tokens.tolist())
