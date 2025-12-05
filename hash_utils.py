import hashlib
import os


def sha256_bytes(data):
    """Compute SHA-256 hash of byte data."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(file_path, chunk_size=8192):
    """Compute SHA-256 hash of a file with efficient chunked reading."""
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def sha256_model(model_path):
    """Compute SHA-256 hash of the model weights file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return sha256_file(model_path)


def verify_file_integrity(file_path, expected_hash):
    """Verify file integrity by comparing SHA-256 hashes."""
    actual_hash = sha256_file(file_path)
    return actual_hash == expected_hash


def get_file_info(file_path):
    """Get file information including size and hash."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    file_hash = sha256_file(file_path)
    
    return {
        "path": file_path,
        "size": file_size,
        "sha256": file_hash
    }