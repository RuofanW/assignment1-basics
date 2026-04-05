import pickle

if __name__ == "__main__":
    with open("tinystories_vocab.pkl", "rb") as f:
        data = pickle.load(f)
        vocab = data["vocab"]
        longest_token = max(vocab.values(), key=len)
        print(f"Longest token: {longest_token}")
        print(f"Length: {len(longest_token)} bytes")
        print(f"Decoded: {longest_token.decode('utf-8', errors='replace')}")