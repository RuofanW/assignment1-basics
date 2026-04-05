import os
import regex as re
from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import pickle


GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
rx = re.compile(GPT2_PAT)
num_processes = 4
num_chunks = 64

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

def worker_pre_tokenize(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    # split by special tokens
    escaped_special_tokens = [re.escape(t) for t in special_tokens]
    pattern = '|'.join(escaped_special_tokens)
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    text_splits = re.split(pattern, text)
    text_splits_filtered = [t for t in text_splits if t]
    # print("length after split: ", len(text_splits_filtered))

    # pre-tokenization and count{a tuple of bytes: count}
    counts = {}
    for piece in tqdm(text_splits_filtered, desc="Pre-tokenizing"):
        for match in rx.finditer(piece):
            pre_token = match.group().encode("utf-8")
            key = tuple(bytes([b]) for b in pre_token)
            if key not in counts:
                counts[key] = 0
            counts[key] += 1
    return counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    print("entering pre-tokenization")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    counts = {}
    with Pool(num_processes) as pool:
        args = [(input_path, start, end) for start, end in chunk_ranges]
        worker = partial(worker_pre_tokenize, special_tokens=special_tokens)
        results = pool.starmap(worker, args)
        for partial_counts in results:
            for key, count in partial_counts.items():
                counts[key] = counts.get(key, 0) + count

    print("finished pre-tokenization")
    
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
    
    total_steps_to_go = vocab_size - len(vocab)
    j = 0
    # BPE main loop
    while len(vocab) < vocab_size:
        if j % 500 == 0:
            print("merged ", j, " steps out of ", total_steps_to_go)

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
        j += 1
        
    
    return vocab, merges

if __name__ == "__main__":
    path_tinystories = "data/TinyStoriesV2-GPT4-train.txt"
    path_owt = "data/owt_train.txt"
    # vocab, merges = train_bpe(path_tinystories, 10000, ['<|endoftext|>'])
    vocab, merges = train_bpe(path_owt, 32000, ['<|endoftext|>'])

    with open("owt_vocab.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

