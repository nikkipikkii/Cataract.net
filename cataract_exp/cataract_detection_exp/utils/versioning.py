import hashlib

def compute_file_hash(path, algo="sha256"):
    hash_fn = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_fn.update(chunk)
    return hash_fn.hexdigest()
