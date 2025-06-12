#!/usr/bin/env python3
"""
🏆 LUNA16 MEDICAL AI DASHBOARD - Duke University AIPI540 Project
Interactive Streamlit Dashboard for Lung Nodule Detection
Deep Learning - Computer Vision Final Project
Author: Evan Moh
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="LUNA16 Lung Cancer Detection - Evan Moh | Duke AIPI540",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #012169;
    margin-bottom: 1rem;
}
.duke-blue {
    color: #012169;
}
.project-info {
    text-align: center;
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 2px solid #012169;
    margin-bottom: 2rem;
    color: #000000;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.project-info h3, .project-info h4, .project-info p {
    color: #000000 !important;
    margin: 0.5rem 0;
}
.duke-blue {
    color: #012169 !important;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #012169;
}
.success-metric {
    border-left-color: #28a745;
}
.warning-metric {
    border-left-color: #ffc107;
}
.danger-metric {
    border-left-color: #dc3545;
}
.model-header {
    background: linear-gradient(90deg, #012169, #1f77b4);
    color: white;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Project data based on your actual results
@st.cache_data
def get_actual_model_data():
    """Your latest actual model performance data"""
    return {
        'Naive Baseline': {
            'froc': 0.0286, 
            'auc': 0.5090, 
            'training_time': 0.1,
            'predicted_positives': 'Heuristic-based',
            'total_samples': 1000,
            'description': 'Simple coordinate-based heuristics',
            'status': '⚠️ BASELINE'
        },
        'Random Forest (Classic ML)': {
            'froc': 0.0429, 
            'auc': 0.6590, 
            'training_time': 2.5,
            'predicted_positives': 'Enhanced features',
            'total_samples': 1000,
            'description': 'Enhanced feature engineering + class balance',
            'status': '⚠️ BASELINE'
        },
        'Simple Effective 3D CNN': {
            'froc': 0.1429, 
            'auc': 0.6562, 
            'training_time': 1.5,
            'predicted_positives': '66 CT patches',
            'total_samples': 1000,
            'description': '3D CNN + ResNet + Transformer + Attention',
            'status': '🥇 BEST PERFORMANCE'
        }
    }

@st.cache_data
def get_dataset_info():
    """Your latest actual dataset information"""
    return {
        'total_candidates': 754975,
        'positive_samples': 1557,
        'negative_samples': 753418,
        'training_dataset': 1000,
        'train_samples': 46,
        'test_samples': 20,
        'ct_patches': 66,
        'positive_ratio': 0.20,  # 200/1000 in training set
        'class_distribution': {'Negative (0)': 800, 'Positive (1)': 200},
        'total_time_minutes': 1.5,
        'gpu_available': True,
        'patch_size': '64³'
    }

@st.cache_data
def generate_training_history_3dcnn():
    """Generate training history based on your actual 3D CNN training"""
    # Your actual training data shows 40 epochs
    epochs = list(range(1, 41))
    
    # Approximate values based on your training output
    train_loss = [2.4563, 0.9171, 0.9667, 0.7979, 0.8106, 0.8707, 0.8772, 0.8039, 0.7860, 0.7459,
                  0.8238, 0.7709, 0.7967, 0.7987, 0.6891, 0.7474, 0.7926, 0.7312, 0.7312, 0.6798,
                  0.7037, 0.8019, 0.7092, 0.7248, 0.7453, 0.7361, 0.7466, 0.6821, 0.6710, 0.6511,
                  0.6887, 0.6780, 0.6853, 0.6964, 0.7043, 0.6760, 0.6696, 0.6667, 0.6799, 0.6965]
    
    val_loss = [0.4656, 0.3087, 0.2635, 0.2355, 0.2227, 0.2132, 0.2115, 0.2059, 0.2134, 0.2103,
                0.2035, 0.2052, 0.2048, 0.2061, 0.2045, 0.2055, 0.2050, 0.2035, 0.2030, 0.2037,
                0.2005, 0.2025, 0.2019, 0.2027, 0.2018, 0.2005, 0.2012, 0.2009, 0.1999, 0.2017,
                0.2008, 0.2020, 0.2042, 0.1997, 0.1993, 0.2013, 0.2014, 0.2022, 0.1985, 0.2018]
    
    # Pad with zeros if needed
    while len(train_loss) < len(epochs):
        train_loss.append(train_loss[-1] * 0.98)
        val_loss.append(val_loss[-1] * 0.99)
    
    train_loss = train_loss[:len(epochs)]
    val_loss = val_loss[:len(epochs)]
    
    # AUC values (approximate from your output)
    train_auc = np.linspace(0.5195, 0.7027, len(epochs))
    val_auc = np.linspace(0.5813, 0.6551, len(epochs))
    
    return pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_auc': train_auc,
        'val_auc': val_auc
    })

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 LUNA16 Lung Nodule Detection</h1>', unsafe_allow_html=True)
    
    # Project info
    st.markdown("""
    <div class="project-info">
        <h3 class="duke-blue" style="color: #012169 !important;">Duke University AIPI540 - Deep Learning Computer Vision</h3>
        <h4 style="color: #000000 !important;">Final Project by <strong>Evan Moh</strong></h4>
        <p style="color: #000000 !important;"><strong>Target:</strong> FROC Score > 0.85 | <strong>Current Status:</strong> 🔄 In Development</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Project Controls")
    st.sidebar.markdown("**Student:** Evan Moh")
    st.sidebar.markdown("**Course:** Duke AIPI540")
    st.sidebar.markdown("**Project:** Lung Nodule Detection")
    
    # Model selection
    model_data = get_actual_model_data()
    model_options = list(model_data.keys())
    selected_model = st.sidebar.selectbox("Select Model Approach", model_options, index=2)
    
    # Dataset info
    dataset_info = get_dataset_info()
    
    st.sidebar.markdown("## 📊 Dataset Overview")
    st.sidebar.write(f"**Total Candidates:** {dataset_info['total_candidates']:,}")
    st.sidebar.write(f"**Positive Samples:** {dataset_info['positive_samples']:,}")
    st.sidebar.write(f"**Test Set:** {dataset_info['test_samples']:,} samples")
    st.sidebar.write(f"**Positive Ratio:** {dataset_info['positive_ratio']:.1%}")
    
    # Three Model Approaches Section
    st.markdown("## 🎯 Three-Model Approach Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="model-header"><h4>1️⃣ Naive Baseline</h4></div>', unsafe_allow_html=True)
        naive_data = model_data['Naive Baseline']
        st.markdown(f"**FROC:** {naive_data['froc']:.4f}")
        st.markdown(f"**AUC:** {naive_data['auc']:.4f}")
        st.markdown(f"**Status:** {naive_data['status']}")
        st.info("Simple baseline approach for comparison")
    
    with col2:
        st.markdown('<div class="model-header"><h4>2️⃣ Random Forest</h4></div>', unsafe_allow_html=True)
        rf_data = model_data['Random Forest (Classic ML)']
        st.markdown(f"**FROC:** {rf_data['froc']:.4f}")
        st.markdown(f"**AUC:** {rf_data['auc']:.4f}")
        st.markdown(f"**Status:** {rf_data['status']}")
        st.info("Classic ML with advanced feature engineering")
    
    with col3:
        st.markdown('<div class="model-header"><h4>3️⃣ Simple Effective 3D CNN</h4></div>', unsafe_allow_html=True)
        cnn_data = model_data['Simple Effective 3D CNN']
        st.markdown(f"**FROC:** {cnn_data['froc']:.4f}")
        st.markdown(f"**AUC:** {cnn_data['auc']:.4f}")
        st.markdown(f"**Status:** {cnn_data['status']}")
        st.info("3D CNN + ResNet + Transformer architecture")
    
    # Current Model Performance
    st.markdown("---")
    st.markdown(f"## 📊 {selected_model} - Detailed Performance")
    
    current_model = model_data[selected_model]
    
    # Performance highlight for best model
    if selected_model == 'Simple Effective 3D CNN':
        st.success("🥇 **BEST PERFORMANCE ACHIEVED!** This model shows significant improvement over baseline approaches.")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        froc_value = current_model['froc']
        target_diff = froc_value - 0.85
        froc_color = "success-metric" if froc_value > 0.85 else "danger-metric"
        st.markdown(f'<div class="metric-card {froc_color}">', unsafe_allow_html=True)
        st.metric("FROC Score", f"{froc_value:.4f}", f"{target_diff:.4f} from target")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        auc_value = current_model['auc']
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("AUC Score", f"{auc_value:.4f}", f"+{(auc_value - 0.5):.4f} vs random")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        training_time = current_model['training_time']
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Time", f"{training_time:.1f} min", "GPU Accelerated")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        pred_pos = current_model['predicted_positives']
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predictions", f"{pred_pos}/{current_model['total_samples']}", "Test Set")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model comparison chart
    st.markdown("## 📈 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FROC comparison
        models = list(model_data.keys())
        froc_scores = [model_data[model]['froc'] for model in models]
        auc_scores = [model_data[model]['auc'] for model in models]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[m.replace(' (Classic ML)', '').replace(' (Deep Learning)', '') for m in models],
            y=froc_scores,
            name='FROC Score',
            marker_color=['#dc3545', '#ffc107', '#012169'],
            text=[f'{score:.4f}' for score in froc_scores],
            textposition='auto',
        ))
        
        # Target line
        fig.add_hline(y=0.85, line_dash="dash", line_color="red", 
                     annotation_text="Target: 0.85", annotation_position="top right")
        
        fig.update_layout(
            title="FROC Performance Comparison",
            xaxis_title="Model Approach",
            yaxis_title="FROC Score",
            height=400,
            showlegend=False,
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AUC comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[m.replace(' (Classic ML)', '').replace(' (Deep Learning)', '') for m in models],
            y=auc_scores,
            name='AUC Score',
            marker_color=['#dc3545', '#ffc107', '#012169'],
            text=[f'{score:.4f}' for score in auc_scores],
            textposition='auto',
        ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Random Baseline", annotation_position="top right")
        
        fig.update_layout(
            title="AUC Performance Comparison",
            xaxis_title="Model Approach",
            yaxis_title="AUC Score",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D CNN Training History (only show if 3D CNN is selected)
    if selected_model == '3D CNN (Deep Learning)':
        st.markdown("## 🧠 3D CNN Training Progress")
        
        training_data = generate_training_history_3dcnn()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curves
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=training_data['epoch'], 
                y=training_data['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=training_data['epoch'], 
                y=training_data['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="3D CNN Training & Validation Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400,
                legend=dict(x=0.7, y=0.9)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # AUC curves
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=training_data['epoch'], 
                y=training_data['train_auc'],
                mode='lines',
                name='Training AUC',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=training_data['epoch'], 
                y=training_data['val_auc'],
                mode='lines',
                name='Validation AUC',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="3D CNN Training & Validation AUC",
                xaxis_title="Epoch",
                yaxis_title="AUC Score",
                height=400,
                legend=dict(x=0.1, y=0.9)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Analysis
    st.markdown("## 📊 Dataset Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔢 Data Statistics")
        st.write(f"**Total Candidates:** {dataset_info['total_candidates']:,}")
        st.write(f"**Competition Sample:** {dataset_info['competition_sample']:,}")
        st.write(f"**Training Set:** {dataset_info['train_samples']:,}")
        st.write(f"**Test Set:** {dataset_info['test_samples']:,}")
        st.write(f"**Test Positives:** {dataset_info['test_positives']:,}")
    
    with col2:
        # Class distribution pie chart
        class_data = dataset_info['class_distribution']
        fig = go.Figure(data=[go.Pie(
            labels=list(class_data.keys()),
            values=list(class_data.values()),
            hole=.3,
            marker_colors=['lightcoral', 'lightblue']
        )])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_text="Class Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 📈 Progress Metrics")
        st.write(f"**Total Training Time:** {dataset_info['total_time_minutes']:.1f} min")
        st.write(f"**Best FROC So Far:** {max([m['froc'] for m in model_data.values()]):.4f}")
        st.write(f"**Best AUC So Far:** {max([m['auc'] for m in model_data.values()]):.4f}")
        st.write(f"**Target Gap:** {0.85 - max([m['froc'] for m in model_data.values()]):.4f}")
    
    # Project Status
    st.markdown("---")
    st.markdown("## 🎯 Project Status & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Completed")
        st.write("✅ Dataset preprocessing and feature engineering")
        st.write("✅ Naive baseline implementation")
        st.write("✅ Random Forest with advanced features")
        st.write("✅ 3D CNN architecture design and training")
        st.write("✅ Initial model evaluation")
    
    with col2:
        st.markdown("### 🔄 In Progress / Next Steps")
        st.write("🔄 Hyperparameter optimization for 3D CNN")
        st.write("🔄 Data augmentation strategies")
        st.write("🔄 Ensemble methods combining all three models")
        st.write("🔄 Advanced 3D CNN architectures (ResNet3D, DenseNet3D)")
        st.write("🎯 **Goal:** Achieve FROC > 0.85")
    
    # Footer
    st.markdown("---")
    current_time = datetime.now()
    st.markdown(f"""
    **Duke University AIPI540 - Deep Learning Computer Vision Project**  
    **Student:** Evan Moh | **Last Updated:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}  
    **Status:** 🔄 Development in Progress | **Target:** FROC > 0.85
    """)

if __name__ == "__main__":
    main()
