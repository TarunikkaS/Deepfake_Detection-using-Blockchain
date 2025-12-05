# Blockchain-Secured Deepfake Detection System

A production-ready pipeline for deepfake detection with blockchain provenance and explainable AI.

## Features

- **Deep Learning**: Xception-based model for deepfake detection
- **Blockchain Provenance**: Immutable logging of detection results
- **Explainable AI**: Grad-CAM visualizations for model interpretability  
- **Web Interface**: Streamlit app for easy interaction
- **IPFS Integration**: Decentralized storage (placeholder implementation)

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your blockchain credentials
   ```

3. **Deploy Smart Contract**
   - Deploy `contracts/DeepfakeProvenance.sol` to your Ethereum network
   - Update `CONTRACT_ADDRESS` and `CONTRACT_ABI` in `.env`

## Usage

### Command Line Interface
```bash
python inference_pipeline.py path/to/image.jpg
```

### Web Interface
```bash
streamlit run app.py
```

### Programmatic Usage
```python
from inference_pipeline import run_inference_pipeline

result = run_inference_pipeline("image.jpg")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}")
print(f"Blockchain TX: {result['tx_hash']}")
```

## Project Structure

```
blockchain_PROJ/
├── models/
│   └── xception_gan_augmented.pth    # Trained model weights
├── contracts/
│   └── DeepfakeProvenance.sol        # Smart contract
├── outputs/
│   └── xai/                          # Grad-CAM visualizations
├── model_inference.py                # Model inference module
├── hash_utils.py                     # SHA-256 utilities
├── xai_gradcam.py                    # Grad-CAM implementation
├── blockchain_client.py              # Web3 blockchain client
├── inference_pipeline.py             # End-to-end pipeline
├── app.py                           # Streamlit web app
├── requirements.txt                 # Python dependencies
└── .env.example                     # Environment template
```

## Architecture

1. **Image Input** → Preprocessing (299×299, normalization)
2. **Model Inference** → Xception model prediction
3. **Hash Generation** → SHA-256 of input image
4. **XAI Generation** → Grad-CAM heatmap
5. **Blockchain Logging** → Immutable record on Ethereum
6. **IPFS Storage** → Decentralized file storage (optional)

## Smart Contract Functions

- `registerModel(name, weightsHash)` - Register a new model
- `logDetection(modelId, mediaHash, label, confidence, mediaCid, xaiCid)` - Log detection result
- `getDetection(detectionId)` - Retrieve detection record
- `getModel(modelId)` - Retrieve model information

## Security Notes

- Never commit private keys to version control
- Use environment variables for sensitive configuration
- Validate all inputs before blockchain transactions
- Consider gas costs for blockchain operations

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Ethereum node (local or remote)
- Solidity ^0.8.20