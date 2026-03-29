import os
import regex as re
from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from functools import partial


GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
rx = re.compile(GPT2_PAT)
num_processes = 4

def load_text(input_path: str) -> str:
    with open(input_path, "r") as f:
        return f.read()


def compute_pairs_single_tuple(tp: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    res: dict[tuple[bytes, bytes], int] = {}
    for i in range(len(tp) - 1):
        k = (tp[i], tp[i + 1])
        res[k] = res.get(k, 0) + 1
    return res


def compute_pairs(tuple_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    res: dict[tuple[bytes, bytes], int] = {}
    for tp, freq in tuple_counts.items():
        local = compute_pairs_single_tuple(tp)
        for k, occ in local.items():
            res[k] = res.get(k, 0) + freq * occ
    return res

def worker_pre_tokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    # split by special tokens
    escaped_special_tokens = [re.escape(t) for t in special_tokens]
    pattern = '|'.join(escaped_special_tokens)
    text_splits = re.split(pattern, text)
    text_splits_filtered = [t for t in text_splits if t]
    print("length after split: ", len(text_splits_filtered))

    # pre-tokenization and count{a tuple of bytes: count}
    counts = {}
    for piece in text_splits_filtered:
        for match in rx.finditer(piece):
            pre_token = match.group().encode("utf-8")
            key = tuple(bytes([b]) for b in pre_token)
            if key not in counts:
                counts[key] = 0
            counts[key] += 1
    return counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    
    with Pool(num_processes) as pool:
        worker = partial(worker_pre_tokenize, special_tokens=special_tokens)
        results = pool.map(worker, chunks)

    counts = {}
    for partial_counts in results:
        for key, count in partial_counts.items():
            counts[key] = counts.get(key, 0) + count 
    
    # print 5 entries from counts
    # for i, (key, count) in enumerate(counts.items()):
    #     if i >= 5:
    #         break
    #     print(f"{key}: {count}")
    # print("--------------------------------")

    # initialize vocab 
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")

    
    merges: list[tuple[bytes, bytes]] = []

    # compute pair stats once initially
    pairs_count = compute_pairs(counts)

    
    # BPE main loop
    while len(vocab) < vocab_size:

        # find the top pair, break ties lexicographically
        best_pair = max(pairs_count, key=lambda p:(pairs_count[p], p))

        # merge the best pair for entries in `counts`
        new_counts_tmp = {}
        
        for k in counts:
            i = 0
            newk = tuple()
            while i  < len(k) - 1:
                if (k[i], k[i + 1]) == best_pair:
                    # newk += (best_pair)
                    newk += (best_pair[0] + best_pair[1],)
                    i += 2
                else:
                    newk += (k[i],)
                    i += 1
            if i == len(k) - 1:
                newk += (k[i],)
            new_counts_tmp[newk] = counts[k] + new_counts_tmp.get(newk, 0)

            if newk != k:
                # incrementally update pairs_count
                local_k = compute_pairs_single_tuple(k)
                for e, occ in local_k.items():
                    if e in pairs_count:
                        pairs_count[e] -= occ * counts[k]
                local_newk = compute_pairs_single_tuple(newk)
                for e, occ in local_newk.items():
                    pairs_count[e] = pairs_count.get(e, 0) + occ * counts[k]

        
        counts = new_counts_tmp

        # add the new pair to vocab and store it in merges
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        
    
    return vocab, merges




    




    














