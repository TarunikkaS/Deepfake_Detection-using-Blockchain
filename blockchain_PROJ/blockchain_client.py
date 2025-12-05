from web3 import Web3
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BlockchainClient:
    def __init__(self):
        self.w3 = None
        self.contract = None
        self.account = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Web3 connection and contract instance."""
        # Get configuration from environment
        rpc_url = os.getenv('WEB3_PROVIDER_URL', 'http://localhost:8545')
        contract_address = os.getenv('CONTRACT_ADDRESS')
        contract_abi = os.getenv('CONTRACT_ABI')
        private_key = os.getenv('PRIVATE_KEY')
        
        if not all([contract_address, contract_abi, private_key]):
            raise ValueError("Missing required environment variables: CONTRACT_ADDRESS, CONTRACT_ABI, PRIVATE_KEY")
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        # Load account from private key
        self.account = self.w3.eth.account.from_key(private_key)
        
        # Initialize contract
        abi = json.loads(contract_abi)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )
    
    def get_account_balance(self):
        """Get account balance in ETH."""
        balance_wei = self.w3.eth.get_balance(self.account.address)
        return self.w3.from_wei(balance_wei, 'ether')
    
    def get_gas_price(self):
        """Get current gas price."""
        return self.w3.eth.gas_price
    
    def log_detection(self, model_id, media_hash, label, confidence, media_cid="", xai_cid=""):
        """
        Log a deepfake detection result to the blockchain.
        
        Args:
            model_id (int): ID of the model used
            media_hash (str): SHA-256 hash of the media file
            label (str): Detection result ("real" or "fake")
            confidence (float): Confidence score (0-1), will be converted to 0-100
            media_cid (str): IPFS CID of the media file
            xai_cid (str): IPFS CID of the XAI explanation
        
        Returns:
            str: Transaction hash
        """
        # Convert confidence to integer percentage
        confidence_int = int(confidence * 100)
        
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            gas_estimate = self.contract.functions.logDetection(
                model_id,
                media_hash,
                label,
                confidence_int,
                media_cid,
                xai_cid
            ).estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = self.contract.functions.logDetection(
                model_id,
                media_hash,
                label,
                confidence_int,
                media_cid,
                xai_cid
            ).build_transaction({
                'chainId': self.w3.eth.chain_id,
                'gas': gas_estimate,
                'gasPrice': self.get_gas_price(),
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt.transactionHash.hex()
            
        except Exception as e:
            raise Exception(f"Failed to log detection to blockchain: {str(e)}")
    
    def get_detection_by_id(self, detection_id):
        """Retrieve detection record by ID."""
        try:
            detection = self.contract.functions.getDetection(detection_id).call()
            return {
                'id': detection[0],
                'model_id': detection[1],
                'media_hash': detection[2],
                'label': detection[3],
                'confidence': detection[4],
                'media_cid': detection[5],
                'xai_cid': detection[6],
                'timestamp': detection[7]
            }
        except Exception as e:
            raise Exception(f"Failed to retrieve detection: {str(e)}")
    
    def get_model_by_id(self, model_id):
        """Retrieve model record by ID."""
        try:
            model = self.contract.functions.getModel(model_id).call()
            return {
                'id': model[0],
                'name': model[1],
                'weights_hash': model[2]
            }
        except Exception as e:
            raise Exception(f"Failed to retrieve model: {str(e)}")
    
    def register_model(self, name, weights_hash):
        """
        Register a new AI model in the blockchain.
        
        Args:
            name (str): Name of the model
            weights_hash (str): SHA-256 hash of the model weights
        
        Returns:
            str: Transaction hash
        """
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            gas_estimate = self.contract.functions.registerModel(
                name,
                weights_hash
            ).estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = self.contract.functions.registerModel(
                name,
                weights_hash
            ).build_transaction({
                'chainId': self.w3.eth.chain_id,
                'gas': gas_estimate,
                'gasPrice': self.get_gas_price(),
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt.transactionHash.hex()
            
        except Exception as e:
            raise Exception(f"Failed to register model: {str(e)}")


def create_blockchain_client():
    """Factory function to create blockchain client."""
    return BlockchainClient()