import streamlit as st
import os
from PIL import Image
import numpy as np
from inference_pipeline import run_inference_pipeline
import plotly.graph_objects as go


def main():
    st.set_page_config(
        page_title="Blockchain-Secured Deepfake Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Blockchain-Secured Deepfake Detection System")
    st.markdown("Upload an image to detect if it's real or artificially generated (deepfake)")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        model_id = st.number_input("Model ID", min_value=1, value=1, help="ID of the model in the blockchain")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, help="Minimum confidence for classification")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run inference button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Running deepfake detection..."):
                    try:
                        # Run the inference pipeline
                        result = run_inference_pipeline(temp_path, model_id)
                        
                        # Store result in session state
                        st.session_state.result = result
                        st.session_state.temp_path = temp_path
                        
                        st.success("Analysis completed!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    with col2:
        st.header("Results")
        
        if 'result' in st.session_state:
            result = st.session_state.result
            
            # Display prediction with color coding
            label = result['label']
            confidence = result['confidence']
            
            if label == "real":
                st.success(f"✅ **REAL** (Confidence: {confidence:.4f})")
                label_color = "green"
            else:
                st.error(f"⚠️ **DEEPFAKE** (Confidence: {confidence:.4f})")
                label_color = "red"
            
            # Confidence meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': label_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence_threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw probabilities
            st.subheader("Raw Probabilities")
            prob_real = result['raw_probabilities'][0]
            prob_fake = result['raw_probabilities'][1]
            
            col_real, col_fake = st.columns(2)
            with col_real:
                st.metric("Real", f"{prob_real:.4f}")
            with col_fake:
                st.metric("Fake", f"{prob_fake:.4f}")
            
            # Display Grad-CAM heatmap
            st.subheader("🎯 Grad-CAM Explanation")
            if os.path.exists(result['gradcam_path']):
                gradcam_image = Image.open(result['gradcam_path'])
                st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
                st.info("The heatmap shows which parts of the image the model focused on for its decision.")
            else:
                st.warning("Grad-CAM visualization not available")
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.write("**Media Hash (SHA-256):**")
                st.code(result['media_hash'])
                
                if result.get('tx_hash'):
                    st.write("**Blockchain Transaction:**")
                    st.code(result['tx_hash'])
                    st.success("✅ Result logged to blockchain")
                else:
                    st.warning("⚠️ Blockchain logging failed")
                
                if result.get('media_cid'):
                    st.write("**IPFS Media CID:**")
                    st.code(result['media_cid'])
                
                if result.get('xai_cid'):
                    st.write("**IPFS XAI CID:**")
                    st.code(result['xai_cid'])
            
            # Clean up temp file after displaying results
            temp_path = st.session_state.get('temp_path')
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🔐 <strong>Blockchain-Secured Deepfake Detection System</strong></p>
            <p>Powered by PyTorch, Ethereum, and IPFS</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()