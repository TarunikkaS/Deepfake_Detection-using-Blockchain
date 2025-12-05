// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DeepfakeProvenance {
    struct Model {
        uint256 id;
        string name;
        string weightsHash;
    }
    
    struct Detection {
        uint256 id;
        uint256 modelId;
        string mediaHash;
        string label;
        uint256 confidence;
        string mediaCid;
        string xaiCid;
        uint256 timestamp;
    }
    
    mapping(uint256 => Model) public models;
    mapping(uint256 => Detection) public detections;
    
    uint256 public nextModelId;
    uint256 public nextDetectionId;
    
    address public owner;
    
    event ModelRegistered(
        uint256 indexed modelId,
        string name,
        string weightsHash
    );
    
    event DetectionLogged(
        uint256 indexed detectionId,
        uint256 indexed modelId,
        string mediaHash,
        string label,
        uint256 confidence,
        string mediaCid,
        string xaiCid,
        uint256 timestamp
    );
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        nextModelId = 1;
        nextDetectionId = 1;
    }
    
    function registerModel(
        string memory name,
        string memory weightsHash
    ) external onlyOwner returns (uint256) {
        require(bytes(name).length > 0, "Model name cannot be empty");
        require(bytes(weightsHash).length > 0, "Weights hash cannot be empty");
        
        uint256 modelId = nextModelId;
        nextModelId++;
        
        models[modelId] = Model({
            id: modelId,
            name: name,
            weightsHash: weightsHash
        });
        
        emit ModelRegistered(modelId, name, weightsHash);
        
        return modelId;
    }
    
    function logDetection(
        uint256 modelId,
        string memory mediaHash,
        string memory label,
        uint256 confidence,
        string memory mediaCid,
        string memory xaiCid
    ) external returns (uint256) {
        require(modelId > 0 && modelId < nextModelId, "Invalid model ID");
        require(bytes(mediaHash).length > 0, "Media hash cannot be empty");
        require(bytes(label).length > 0, "Label cannot be empty");
        require(confidence <= 100, "Confidence must be between 0 and 100");
        require(
            keccak256(abi.encodePacked(label)) == keccak256(abi.encodePacked("real")) ||
            keccak256(abi.encodePacked(label)) == keccak256(abi.encodePacked("fake")),
            "Label must be 'real' or 'fake'"
        );
        
        uint256 detectionId = nextDetectionId;
        nextDetectionId++;
        
        detections[detectionId] = Detection({
            id: detectionId,
            modelId: modelId,
            mediaHash: mediaHash,
            label: label,
            confidence: confidence,
            mediaCid: mediaCid,
            xaiCid: xaiCid,
            timestamp: block.timestamp
        });
        
        emit DetectionLogged(
            detectionId,
            modelId,
            mediaHash,
            label,
            confidence,
            mediaCid,
            xaiCid,
            block.timestamp
        );
        
        return detectionId;
    }
    
    function getModel(uint256 modelId) external view returns (
        uint256 id,
        string memory name,
        string memory weightsHash
    ) {
        require(modelId > 0 && modelId < nextModelId, "Invalid model ID");
        
        Model memory model = models[modelId];
        return (model.id, model.name, model.weightsHash);
    }
    
    function getDetection(uint256 detectionId) external view returns (
        uint256 id,
        uint256 modelId,
        string memory mediaHash,
        string memory label,
        uint256 confidence,
        string memory mediaCid,
        string memory xaiCid,
        uint256 timestamp
    ) {
        require(detectionId > 0 && detectionId < nextDetectionId, "Invalid detection ID");
        
        Detection memory detection = detections[detectionId];
        return (
            detection.id,
            detection.modelId,
            detection.mediaHash,
            detection.label,
            detection.confidence,
            detection.mediaCid,
            detection.xaiCid,
            detection.timestamp
        );
    }
    
    function getDetectionsByModel(uint256 modelId) external view returns (uint256[] memory) {
        require(modelId > 0 && modelId < nextModelId, "Invalid model ID");
        
        uint256[] memory result = new uint256[](nextDetectionId - 1);
        uint256 count = 0;
        
        for (uint256 i = 1; i < nextDetectionId; i++) {
            if (detections[i].modelId == modelId) {
                result[count] = i;
                count++;
            }
        }
        
        // Resize array to actual count
        uint256[] memory trimmedResult = new uint256[](count);
        for (uint256 j = 0; j < count; j++) {
            trimmedResult[j] = result[j];
        }
        
        return trimmedResult;
    }
    
    function getTotalModels() external view returns (uint256) {
        return nextModelId - 1;
    }
    
    function getTotalDetections() external view returns (uint256) {
        return nextDetectionId - 1;
    }
    
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        owner = newOwner;
    }
}