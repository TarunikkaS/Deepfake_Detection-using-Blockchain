import os
from model_inference import predict_image
from hash_utils import sha256_file
from xai_gradcam import save_gradcam_visualization
from blockchain_client import create_blockchain_client


def run_inference_pipeline(image_path, model_id=1, model_path="models/xception_gan_augmented.pth"):
    """
    End-to-end inference pipeline for deepfake detection.
    
    Args:
        image_path (str): Path to the input image
        model_id (int): ID of the model for blockchain logging
        model_path (str): Path to the trained model
    
    Returns:
        dict: Complete inference results
    """
    try:
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Step 1: Run inference
        print("Running deepfake detection...")
        inference_result = predict_image(image_path, model_path)
        label = inference_result["label"]
        confidence = inference_result["confidence"]
        
        # Step 2: Compute SHA-256 hash of the image
        print("Computing image hash...")
        media_hash = sha256_file(image_path)
        
        # Step 3: Generate Grad-CAM visualization
        print("Generating Grad-CAM visualization...")
        gradcam_path = save_gradcam_visualization(image_path, model_path)
        
        # Step 4: IPFS upload placeholders (to be implemented)
        media_cid = ""  # TODO: Upload to IPFS
        xai_cid = ""    # TODO: Upload Grad-CAM to IPFS
        
        # Step 5: Log to blockchain
        print("Logging to blockchain...")
        try:
            blockchain_client = create_blockchain_client()
            tx_hash = blockchain_client.log_detection(
                model_id=model_id,
                media_hash=media_hash,
                label=label,
                confidence=confidence,
                media_cid=media_cid,
                xai_cid=xai_cid
            )
            print(f"Transaction hash: {tx_hash}")
        except Exception as e:
            print(f"Blockchain logging failed: {str(e)}")
            tx_hash = None
        
        # Step 6: Return structured results
        result = {
            "label": label,
            "confidence": confidence,
            "raw_probabilities": inference_result["raw_probabilities"],
            "media_hash": media_hash,
            "gradcam_path": gradcam_path,
            "tx_hash": tx_hash,
            "media_cid": media_cid,
            "xai_cid": xai_cid
        }
        
        print("Pipeline completed successfully!")
        return result
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        raise


def batch_inference(image_paths, model_id=1, model_path="models/xception_gan_augmented.pth"):
    """
    Run inference pipeline on multiple images.
    
    Args:
        image_paths (list): List of image file paths
        model_id (int): ID of the model for blockchain logging
        model_path (str): Path to the trained model
    
    Returns:
        list: Results for each image
    """
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
        try:
            result = run_inference_pipeline(image_path, model_id, model_path)
            results.append(result)
        except Exception as e:
            print(f"Failed to process {image_path}: {str(e)}")
            results.append({"error": str(e), "image_path": image_path})
    
    return results


def print_results(result):
    """Print inference results in a formatted way."""
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION RESULTS")
    print("="*50)
    print(f"Label: {result['label'].upper()}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Media Hash: {result['media_hash']}")
    print(f"Grad-CAM Path: {result['gradcam_path']}")
    
    if result.get('tx_hash'):
        print(f"Blockchain TX: {result['tx_hash']}")
    else:
        print("Blockchain TX: Failed")
    
    print("="*50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python inference_pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        result = run_inference_pipeline(image_path)
        print_results(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)