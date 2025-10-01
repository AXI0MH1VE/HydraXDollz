# hydra_persistence.py
# A High-Performance, Versioned Model Persistence System for AI Terminals
# MIT License, The HYDRA Project

import os
import json
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple

# Using safetensors for secure and fast model serialization
# Ensure you have the library installed: pip install safetensors
from safetensors.numpy import save_file, load_file
import numpy as np

class ModelPersistenceManager:
    """
    Manages the loading, saving, and versioning of AI models for the HYDRA AI Terminal.
    This system is designed for high performance, data integrity, and parallel operations.
    It is a utility library and has no autonomous capabilities.
    """

    def __init__(self, base_storage_path: str = "./hydra_models"):
        """
        Initializes the persistence manager.

        Args:
            base_storage_path (str): The root directory where models will be stored.
        """
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_path / "manifest.json"
        self._load_manifest()
        print(f"HYDRA Persistence Manager initialized at: {self.base_path.resolve()}")

    def _load_manifest(self):
        """Loads the model manifest file, creating it if it doesn't exist."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"models": {}}
            self._save_manifest()

    def _save_manifest(self):
        """Saves the current state of the manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=4)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculates the SHA256 checksum of a file for integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def save_model(self, model_name: str, version: str, model_data: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None):
        """
        Saves a model to the storage directory with versioning and a checksum.

        Args:
            model_name (str): The name of the model (e.g., 'core-reasoner').
            version (str): The version identifier (e.g., '1.0.2').
            model_data (Dict[str, np.ndarray]): The model weights and tensors.
            metadata (Optional[Dict[str, Any]]): Optional metadata to store with the model.
        """
        start_time = time.perf_counter()
        model_id = f"{model_name}:{version}"
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = model_dir / "model.safetensors"
        
        # Save the model tensors
        save_file(model_data, file_path)
        
        checksum = self._calculate_checksum(file_path)
        
        # Update manifest
        manifest_entry = {
            "path": str(file_path.relative_to(self.base_path)),
            "checksum": checksum,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 4),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {}
        }
        
        self.manifest["models"][model_id] = manifest_entry
        self._save_manifest()
        
        duration = time.perf_counter() - start_time
        print(f"Successfully saved model '{model_id}'. Took {duration:.4f}s.")

    def load_model(self, model_name: str, version: str = "latest") -> Optional[Dict[str, np.ndarray]]:
        """
        Loads a model from storage, verifying its integrity via checksum.

        Args:
            model_name (str): The name of the model to load.
            version (str): The specific version to load, or 'latest' to load the highest semantic version.

        Returns:
            Optional[Dict[str, np.ndarray]]: The loaded model data, or None if not found or corrupt.
        """
        start_time = time.perf_counter()
        if version == "latest":
            model_id = self._find_latest_version(model_name)
            if not model_id:
                print(f"Error: No versions found for model '{model_name}'.")
                return None
        else:
            model_id = f"{model_name}:{version}"

        if model_id not in self.manifest["models"]:
            print(f"Error: Model '{model_id}' not found in manifest.")
            return None
            
        entry = self.manifest["models"][model_id]
        file_path = self.base_path / entry["path"]
        
        if not file_path.exists():
            print(f"Error: Model file for '{model_id}' not found at expected path: {file_path}")
            return None
            
        # 1. Verify integrity
        print(f"Verifying integrity for '{model_id}'...")
        actual_checksum = self._calculate_checksum(file_path)
        if actual_checksum != entry["checksum"]:
            print(f"CRITICAL: Checksum mismatch for model '{model_id}'. File may be corrupt!")
            print(f"  Expected: {entry['checksum']}")
            print(f"  Actual:   {actual_checksum}")
            return None
        
        # 2. Load model
        print(f"Loading model '{model_id}'...")
        model_data = load_file(file_path)
        
        duration = time.perf_counter() - start_time
        print(f"Successfully loaded model '{model_id}'. Took {duration:.4f}s.")
        return model_data

    def list_models(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Lists all available models and their versions from the manifest.

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: A list of tuples, each containing
            (model_name, version, metadata).
        """
        model_list = []
        for model_id, entry in self.manifest["models"].items():
            name, version = model_id.split(":", 1)
            model_list.append((name, version, entry))
        return sorted(model_list, key=lambda x: (x[0], x[1]))

    def _find_latest_version(self, model_name: str) -> Optional[str]:
        """Finds the latest semantic version for a given model name."""
        try:
            from packaging.version import parse as parse_version
        except ImportError:
            print("Warning: 'packaging' library not found. 'latest' version resolution may be basic. `pip install packaging`")
            # Basic fallback if packaging is not installed
            versions = [model_id for model_id in self.manifest["models"] if model_id.startswith(f"{model_name}:")]
            return sorted(versions, reverse=True)[0] if versions else None

        versions = [
            model_id for model_id in self.manifest["models"]
            if model_id.startswith(f"{model_name}:")
        ]
        if not versions:
            return None
        
        latest_version = sorted(versions, key=lambda v: parse_version(v.split(':', 1)[1]), reverse=True)
        return latest_version[0]

    def load_models_parallel(self, model_specs: List[Tuple[str, str]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Loads multiple models in parallel using a thread pool.

        Args:
            model_specs (List[Tuple[str, str]]): A list of (model_name, version) tuples to load.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: A dictionary where keys are model_ids and values
            are the loaded model data.
        """
        print(f"Starting parallel load for {len(model_specs)} models...")
        start_time = time.perf_counter()
        loaded_models = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_model = {
                executor.submit(self.load_model, name, ver): f"{name}:{ver}"
                for name, ver in model_specs
            }
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    model_data = future.result()
                    if model_data:
                        loaded_models[model_id] = model_data
                except Exception as exc:
                    print(f"Model '{model_id}' generated an exception during parallel load: {exc}")
        
        duration = time.perf_counter() - start_time
        print(f"Parallel load complete. Loaded {len(loaded_models)} models in {duration:.4f}s.")
        return loaded_models

# --- System Initialization & Live Simulation ---
if __name__ == "__main__":

    def create_micro_regressor(input_dim: int, output_dim: int) -> Dict[str, np.ndarray]:
        """Creates a simple linear regression model."""
        return {
            'weights': np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01,
            'bias': np.zeros(output_dim, dtype=np.float32),
        }

    def create_micro_vectorizer(vocab_size: int, embed_dim: int) -> Dict[str, np.ndarray]:
        """Creates a simple embedding layer model."""
        return {
            'embedding_matrix': np.random.randn(vocab_size, embed_dim).astype(np.float32)
        }

    print("--- HYDRA Persistence Manager: Live Simulation ---")
    
    # 1. Initialize the manager in a dedicated simulation directory
    manager = ModelPersistenceManager(base_storage_path="./hydra_live_simulation")

    # 2. Create and save two distinct, functional micro-models
    print("\n[SIM] Fabricating and saving foundational micro-models...")
    regressor_model_v1 = create_micro_regressor(input_dim=512, output_dim=1)
    vectorizer_model_v1 = create_micro_vectorizer(vocab_size=1000, embed_dim=128)
    
    manager.save_model(
        model_name="alpha-regressor", 
        version="1.0.0", 
        model_data=regressor_model_v1, 
        metadata={"description": "A foundational linear regression model.", "input_shape": 512, "output_shape": 1}
    )
    
    manager.save_model(
        model_name="genesis-vectorizer", 
        version="1.0.0", 
        model_data=vectorizer_model_v1, 
        metadata={"description": "A foundational word embedding model.", "vocabulary_size": 1000, "embedding_dimension": 128}
    )

    # 3. Create and save an updated version of the regressor
    regressor_model_v2 = create_micro_regressor(input_dim=512, output_dim=4) # Changed output dimension
    manager.save_model(
        model_name="alpha-regressor", 
        version="1.1.0", 
        model_data=regressor_model_v2, 
        metadata={"description": "Updated regressor to support 4 outputs.", "input_shape": 512, "output_shape": 4}
    )

    # 4. List all available models to verify they were saved
    print("\n[SIM] Listing all models in storage...")
    all_models = manager.list_models()
    for name, version, data in all_models:
        print(f"  - Found: {name}:{version} (Size: {data['size_mb']} MB)")

    # 5. Load the latest version of the regressor to see version resolution
    print("\n[SIM] Loading latest version of 'alpha-regressor'...")
    latest_regressor = manager.load_model("alpha-regressor", "latest")
    if latest_regressor:
        print("  Latest 'alpha-regressor' loaded successfully.")
        # We can check the shape of the bias to confirm it's the 4-output version 1.1.0
        print(f"  Verified bias shape: {latest_regressor['bias'].shape}")

    # 6. Demonstrate a parallel load of the foundational models
    print("\n[SIM] Demonstrating parallel load of all models...")
    models_to_load = [("alpha-regressor", "1.1.0"), ("genesis-vectorizer", "1.0.0")]
    parallel_results = manager.load_models_parallel(models_to_load)
    print(f"  Successfully loaded models in parallel: {list(parallel_results.keys())}")
    
    print("\n--- Live Simulation Complete ---")