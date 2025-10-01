# HYDRA Persistence System

A High-Performance, Versioned Model Persistence System for AI. Built for speed, integrity, and scalability, HYDRA is a lightweight utility for any serious AI developer who needs to manage local models efficiently.

This system provides a robust solution for saving, loading, and versioning AI models (e.g., weights, embeddings) directly on your machine. It is designed as a pure utility library with no autonomous capabilities, ensuring it remains fully under developer control.

## Key Features

- **Fast & Secure Serialization:** Uses `.safetensors` for fast, secure, and zero-copy model loading.
- **Built-in Versioning:** Manage multiple versions of a model (e.g., `alpha-regressor:1.0.0`, `alpha-regressor:1.1.0`) with 'latest' version resolution.
- **Data Integrity:** Automatically calculates and verifies SHA256 checksums on save and load to prevent data corruption.
- **Parallel Loading:** A built-in thread pool executor allows for loading multiple models simultaneously to saturate disk I/O and reduce wait times.
- **Rich Manifest:** Keeps track of all models, versions, file sizes, creation dates, and custom metadata in a human-readable `manifest.json`.
- **Lightweight & Pure Python:** No heavy dependencies, easy to integrate into any project.

## Installation

First, ensure you have the necessary dependencies installed:

```bash
pip install numpy safetensors "packaging>=21.0"
```

Then, simply add the `hydra_persistence.py` file to your project.

## Quick Start

Here's how to use the `ModelPersistenceManager` in your project.

```python
from hydra_persistence import ModelPersistenceManager
import numpy as np

# 1. Initialize the manager
manager = ModelPersistenceManager(base_storage_path="./my_ai_models")

# 2. Create and save a new model version
my_model_weights = {'weights': np.random.rand(128, 256), 'bias': np.random.rand(256)}
manager.save_model(
    model_name="my-cool-model", 
    version="1.0.0", 
    model_data=my_model_weights,
    metadata={"description": "My first model version."}
)

# 3. Load the latest version of your model
print("Loading the latest version...")
loaded_model = manager.load_model("my-cool-model", version="latest")

if loaded_model:
    print("Model loaded successfully!")
    print(f"Model keys: {list(loaded_model.keys())}")

# 4. List all models in storage
print("\nAll available models:")
for name, version, data in manager.list_models():
    print(f"  - {name}:{version}")
```

## Advanced Features

### Parallel Model Loading

Load multiple models simultaneously to maximize performance:

```python
# Load multiple models in parallel
models_to_load = [("model-a", "1.0.0"), ("model-b", "2.1.0")]
loaded_models = manager.load_models_parallel(models_to_load)

for model_id, model_data in loaded_models.items():
    print(f"Loaded {model_id} with keys: {list(model_data.keys())}")
```

### Version Management

The system supports semantic versioning with automatic "latest" resolution:

```python
# Save multiple versions
manager.save_model("my-model", "1.0.0", model_v1)
manager.save_model("my-model", "1.1.0", model_v1_1)  
manager.save_model("my-model", "2.0.0", model_v2)

# Load the latest version automatically
latest = manager.load_model("my-model", "latest")  # Loads 2.0.0
```

### Data Integrity

Every model is protected with SHA256 checksums:

```python
# If a file is corrupted, the system will detect it
loaded_model = manager.load_model("my-model", "1.0.0")
# Output: CRITICAL: Checksum mismatch for model 'my-model:1.0.0'. File may be corrupt!
```

## Live Simulation Demo

Run the built-in demonstration to see HYDRA in action:

```bash
python hydra_persistence.py
```

This will create sample models, demonstrate version management, and show parallel loading capabilities.

## Use Cases

- **Model Development:** Version your experimental models during development
- **Production Deployment:** Reliable model storage with integrity verification
- **Model Serving:** Fast parallel loading of multiple models for inference
- **Research:** Organize and track different model architectures and experiments

## Storage Structure

HYDRA organizes models in a clean directory structure:

```
hydra_models/
├── manifest.json              # Model registry with metadata
├── alpha-regressor/
│   ├── 1.0.0/
│   │   └── model.safetensors
│   └── 1.1.0/
│       └── model.safetensors
└── genesis-vectorizer/
    └── 1.0.0/
        └── model.safetensors
```

## Security & Safety

- Uses `safetensors` format to prevent arbitrary code execution
- SHA256 checksums ensure data integrity
- No network access or external dependencies beyond core libraries
- Pure utility library with no autonomous behavior

## Contributing

Contributions are welcome! If you have ideas for new features or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.