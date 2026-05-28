import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        embeddings_list_for_the_sequence_of_token_ids = []
        for tok_id in token_ids:
            embeddings_list_for_the_sequence_of_token_ids.append(embeddings[tok_id])
        return np.round(embeddings_list_for_the_sequence_of_token_ids,5)
        

