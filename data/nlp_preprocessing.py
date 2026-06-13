import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        unique_words = set()
        for sentence in positive + negative:
            for word in sentence.split():
                unique_words.add(word)
        word_to_id = {word: idx + 1 for idx, word in enumerate(sorted(unique_words))}
        encoded_positive = [[word_to_id[word] for word in sentence.split()] for sentence in positive]
        encoded_negative = [[word_to_id[word] for word in sentence.split()] for sentence in negative]
        all_encoded = encoded_positive + encoded_negative
        padded_sequences = nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in all_encoded], batch_first=True)
        return padded_sequences.float()

