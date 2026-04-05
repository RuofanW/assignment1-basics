import regex as re
from typing import Iterable, Generator
import pickle


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.bytes_to_index: dict[bytes, int] = {v: k for k, v in vocab.items()}

        self.special_token_regex = re.compile(f"({"|".join(re.escape(t) for t in self.special_tokens)})")
        self._add_special_tokens_to_vocab_if_needed()

        GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.gpt2_rx = re.compile(GPT2_PAT)

        self.merge_rank: dict[tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(merges)}
    
    def _add_special_tokens_to_vocab_if_needed(self):
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.bytes_to_index:
                self.vocab[len(self.vocab)] = st_bytes
                self.bytes_to_index[st_bytes] = len(self.vocab) - 1

    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # match the training outputs used earlier (pickle dump)
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        
        # first split the text by special tokens, to avoid they are split by the GPT-2 pre-tokenizer
        if self.special_tokens:
            special_token_splits = [p for p in self.special_token_regex.split(text) if p]
        else:
            special_token_splits = [text]
        # then for each split from the above step, do GPT-2 pre-tokenization
        final_pre_tokens = []
        for item in special_token_splits:
            if item in self.special_tokens:
                final_pre_tokens.append(item)
            else:
                sub_tokens = re.findall(self.gpt2_rx, item)
                final_pre_tokens.extend(sub_tokens)
        # then for each pre-token, we do BPE encoding, store byte seqs here rather than raw str
        # Output: [[b'ab', b'c'], [b'<|endoftext|>']]
        tokens_ls = []
        for pt in final_pre_tokens:
            if pt in self.special_tokens:
                tokens_ls.append([pt.encode("utf-8")])
            else:
                bytes_seq = [bytes([b]) for b in pt.encode("utf-8")]
                post_merge_seq = self._merge(bytes_seq)
                tokens_ls.append(post_merge_seq)

        #final encode list
        res = []
        for tokens in tokens_ls:
            res.extend([self.bytes_to_index[tk] for tk in tokens])
        return res
    
    def _merge(self, bytes_seq: list[bytes]) -> list[bytes]:
        """
        perform BPE merge, return a list of merged bytes, e.g. [b'a', b'b', b'c'] -> [b'ab', b'c']
        requirement: in the same order of creation
        """
        if len(bytes_seq) < 2:
            return bytes_seq
        
        while True: # merge until we can't
            pairs = [(bytes_seq[i], bytes_seq[i + 1]) for i in range(len(bytes_seq) - 1)]
            if len(pairs) == 0:
                break
            pair_to_merge = min(pairs, key = lambda k: self.merge_rank.get(k, float('inf')))
            if pair_to_merge not in self.merge_rank: # nothing to merge anymore
                break
            new_bytes_seq = []
            i = 0
            while i < len(bytes_seq) - 1:
                if (bytes_seq[i], bytes_seq[i + 1]) == pair_to_merge:
                    new_bytes_seq.append(bytes_seq[i] + bytes_seq[i + 1])
                    i += 1
                else:
                    new_bytes_seq.append(bytes_seq[i])
                i += 1
            if i == len(bytes_seq) - 1:
                new_bytes_seq.append(bytes_seq[i])
            bytes_seq = new_bytes_seq[:]

        return bytes_seq
    
    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int, None, None]:
        """
        Lazily encodes an iterable of strings into a stream of token IDs.
        Yields IDs one by one to keep memory usage low.
        """
        for text in iterable:
            # self.encode(text) returns a list[int]
            # 'yield from' flattens the list and yields each ID individually
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join([self.vocab[id] for id in ids])
        return full_bytes.decode("utf-8", errors="replace")

