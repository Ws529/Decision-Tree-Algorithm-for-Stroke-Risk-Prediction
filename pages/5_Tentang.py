import streamlit as st
import pandas as pd

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="About - Stroke Prediction",
    page_icon="ℹ️",
    layout="wide"
)

# =====================================================================
# HEADER
# =====================================================================
st.markdown('<p style="font-size:2.5rem; font-weight:bold; color:#1f77b4; text-align:center;">About the Project</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Complete information about the stroke prediction system</p>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# PROJECT OVERVIEW
# =====================================================================
st.markdown("## Project Description")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Stroke Prediction System using Decision Tree
    
    A **Machine Learning** based stroke risk prediction system using the **Decision Tree** algorithm. 
    This project aims to help early detection of stroke risk in patients based on their health data 
    and personal characteristics.
    
    **Background:**
    
    Stroke is a leading cause of death and disability worldwide. Early detection of stroke risk factors 
    is crucial for prevention and timely medical intervention. By using machine learning, we can identify 
    patterns and risk factors that might not be visible manually.
    
    **Project Objectives:**
    
    1. Build an accurate stroke prediction model using Decision Tree
    2. Analyze factors that influence stroke risk
    3. Provide an interactive tool for stroke risk screening
    4. Provide informative and easy-to-understand data visualization
    """)

with col2:
    st.info("""
    ### Key Stats
    
    **Dataset:**
    - 5,110 patient records
    - 11 prediction features
    - 1 target variable
    
    **Model:**
    - Algorithm: Decision Tree
    - Accuracy: ~95%
    - AUC Score: 0.942
    
    **Tech Stack:**
    - Python 3.x
    - Scikit-learn
    - Streamlit
    - Plotly
    - Pandas & NumPy
    """)

st.markdown("---")

# =====================================================================
# ALGORITHM
# =====================================================================
st.markdown("## Algorithm: Decision Tree")

tab1, tab2, tab3 = st.tabs(["Explanation", "Pros & Cons", "How It Works"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### What is Decision Tree?
        
        **Decision Tree** is a supervised learning algorithm used for classification and regression.
        The algorithm creates a tree-like decision structure, where:
        
        - **Root Node**: Topmost node (root)
        - **Internal Nodes**: Decision nodes
        - **Branches**: Outcomes of decisions
        - **Leaf Nodes**: Final results (predictions)
        
        ### Basic Concepts:
        
        Decision Tree works by **splitting the dataset** into smaller subsets based on the 
        **most informative features**. This process is repeated recursively until a stopping condition 
        is reached (e.g., maximum depth or minimum samples).
        """)
    
    with col2:
        st.markdown("""
        ### Split Criteria:
        
        **1. Gini Impurity**
        ```
        Gini = 1 - Sum(pi^2)
        ```
        - Measures probability of misclassification
        - Lower = better
        
        **2. Entropy (Information Gain)**
        ```
        Entropy = -Sum(pi * log2(pi))
        ```
        - Measures uncertainty/disorder
        - Information Gain = entropy reduction
        
        ### Hyperparameters:
        
        - `max_depth`: Maximum tree depth
        - `min_samples_split`: Minimum samples for split
        - `min_samples_leaf`: Minimum samples in leaf
        - `criterion`: Split function (gini/entropy)
        """)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        ### Pros of Decision Tree
        
        1. **Easy to Understand**
           - Intuitive visualization
           - Interpretable for non-technical users
        
        2. **No Scaling Required**
           - Not affected by feature scaling
           - Works with categorical data
        
        3. **Handle Non-linear**
           - Can capture non-linear patterns
           - No distribution assumptions
        
        4. **Feature Importance**
           - Shows which features are important
           - Helps feature selection
        
        5. **Fast**
           - Quick training and prediction
           - Suitable for real-time prediction
        
        6. **Multi-output**
           - Can handle multi-class classification
           - Can be used for regression
        """)
    
    with col2:
        st.warning("""
        ### Cons of Decision Tree
        
        1. **Overfitting**
           - Prone to overfit if too deep
           - Requires pruning/regularization
        
        2. **Instability**
           - Sensitive to data changes
           - Trees can differ with slightly different data
        
        3. **Bias on Imbalanced**
           - Tends to bias toward majority class
           - Needs special handling (class_weight)
        
        4. **Global Optimum**
           - Greedy algorithm (local optimum)
           - Does not guarantee global optimum
        
        5. **Linear Relationships**
           - Less effective for linear relationships
           - Requires many splits for linear patterns
        
        **Solutions:**
        - Random Forest (ensemble)
        - Gradient Boosting
        - Cross-validation
        - Hyperparameter tuning
        """)

with tab3:
    st.markdown("""
    ### How Decision Tree Works
    
    #### Training Steps:
    
    1. **Start with Root Node**
       - All data at root
       - Choose best feature for split
    
    2. **Split Data**
       - Calculate Gini/Entropy for each feature
       - Choose feature with highest Information Gain
       - Divide data into subsets
    
    3. **Recursive Splitting**
       - Repeat process for each subset
       - Create internal nodes and branches
    
    4. **Stopping Criteria**
       - Max depth reached
       - Min samples reached
       - No Information Gain
       - All data in node same class
    
    5. **Create Leaf Nodes**
       - Assign class label
       - Majority class in node
    
    #### Prediction Steps:
    
    1. Start from root
    2. Evaluate condition at each node
    3. Follow branch based on condition
    4. Continue until leaf node
    5. Return class in leaf node
    
    #### Example Flow:
    
    ```
    Root: Age > 60?
    ├── No -> BMI > 30?
    │   ├── No -> No Stroke (Leaf)
    │   └── Yes -> Hypertension?
    │       ├── No -> No Stroke (Leaf)
    │       └── Yes -> Stroke (Leaf)
    └── Yes -> Heart Disease?
        ├── No -> Glucose > 140?
        │   ├── No -> No Stroke (Leaf)
        │   └── Yes -> Stroke (Leaf)
        └── Yes -> Stroke (Leaf)
    ```
    """)

st.markdown("---")

# =====================================================================
# DATASET
# =====================================================================
st.markdown("## Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Healthcare Stroke Prediction Dataset
    
    The dataset used is the **Healthcare Stroke Prediction Dataset** containing health information 
    from 5,110 patients with various characteristics and medical history.
    
    **Data Source:**
    - Public dataset for research
    - Data has been anonymized
    - Does not contain sensitive personal information
    
    **Dataset Characteristics:**
    - **Total Records**: 5,110 patients
    - **Total Features**: 11 features + 1 target
    - **Missing Values**: BMI (201 missing)
    - **Imbalanced**: 95% no stroke, 5% stroke
    - **Data Types**: Numerical and categorical
    """)
    
    st.markdown("""
    ### Dataset Features:
    
    | # | Feature | Type | Description |
    |---|---------|------|-------------|
    | 1 | gender | Categorical | Gender (Male/Female/Other) |
    | 2 | age | Numerical | Patient age (0-82 years) |
    | 3 | hypertension | Binary | Hypertension (0=No, 1=Yes) |
    | 4 | heart_disease | Binary | Heart disease (0=No, 1=Yes) |
    | 5 | ever_married | Binary | Marital status (Yes/No) |
    | 6 | work_type | Categorical | Work type |
    | 7 | Residence_type | Binary | Residence type (Urban/Rural) |
    | 8 | avg_glucose_level | Numerical | Glucose level (mg/dL) |
    | 9 | bmi | Numerical | Body Mass Index |
    | 10 | smoking_status | Categorical | Smoking status |
    | 11 | **stroke** | **Binary** | **Target: Stroke (0/1)** |
    """)

with col2:
    st.info("""
    ### Dataset Statistics
    
    **Target Distribution:**
    - No Stroke: 4,861 (95.1%)
    - Stroke: 249 (4.9%)
    
    **Age:**
    - Average: 43.2 years
    - Min: 0.08 years
    - Max: 82 years
    
    **BMI:**
    - Average: 28.9
    - Missing: 201 (3.9%)
    
    **Glucose:**
    - Average: 106.1 mg/dL
    - Range: 55-272 mg/dL
    
    **Health Conditions:**
    - Hypertension: 9.7%
    - Heart Disease: 5.4%
    """)

st.markdown("---")

# =====================================================================
# TOOLS & TECH
# =====================================================================
st.markdown("## Tools & Technologies")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Python Libraries
    
    **Data Processing:**
    - Pandas
    - NumPy
    
    **Machine Learning:**
    - Scikit-learn
    - Joblib
    
    **Visualization:**
    - Plotly
    - Matplotlib
    - Seaborn
    """)

with col2:
    st.markdown("""
    ### Development Tools
    
    **IDE & Notebook:**
    - Jupyter Notebook
    - VS Code
    
    **Web Framework:**
    - Streamlit
    
    **Version Control:**
    - Git
    - GitHub
    """)

with col3:
    st.markdown("""
    ### Deployment
    
    **Platform:**
    - Streamlit Cloud
    - Heroku (optional)
    
    **Requirements:**
    - Python 3.8+
    - pip/conda
    
    **Model Files:**
    - .pkl format
    - Joblib serialization
    """)

st.markdown("---")

# =====================================================================
# TEAM
# =====================================================================
st.markdown("## Development Team")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>Ferly Ardiansyah</h3>
        <p><strong>ID:</strong> 312310448</p>
        <p>Model Development & Evaluation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>Bayu Aji Yuwono</h3>
        <p><strong>ID:</strong> 312310492</p>
        <p>Algorithm Implementation</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;'>
        <h3>Wawan suwandi</h3>
        <p><strong>ID:</strong> 312310457</p>
        <p>UI/UX & Deployment</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# FLOWCHART & PROGRAM FLOW
# =====================================================================
st.markdown("## Flowchart & Program Flow")

tab1, tab2, tab3 = st.tabs(["Training Flow", "App Flow", "Prediction Flow"])

with tab1:
    st.markdown("""
    ### Model Training Flow (Jupyter Notebook)
    
    ```
    START
      |
      V
    +---------------------+
    | 1. Load Dataset     | <-- healthcare-dataset-stroke-data.csv
    |    5,110 x 12       |
    +----------+----------+
               |
               V
    +---------------------+
    | 2. Data Analysis    |
    |    - Preview        |
    |    - Statistics     |
    |    - Missing values |
    +----------+----------+
               |
               V
    +---------------------+
    | 3. EDA              |
    |    15 Visualizations|
    +----------+----------+
               |
               V
    +---------------------+
    | 4. Preprocessing    |
    |    - Drop ID        |
    |    - Fill NA        |
    |    - Encoding       |
    |    - Scaling        |
    |    - Train-Test     |
    +----------+----------+
               |
               V
    +---------------------+
    | 5. Training         |
    |    - Baseline       |
    |    - Grid Search    |
    |    - Best Model     |
    +----------+----------+
               |
               V
    +---------------------+
    | 6. Evaluation       |
    |    Accuracy: ~95%   |
    |    AUC: 0.942       |
    +----------+----------+
               |
               V
    +---------------------+
    | 7. Save Models      |
    |    - .pkl files     |
    +----------+----------+
               |
               V
             [END]
        Models Ready!
    ```
    """)
    
    st.info("""
    **Output Files:**
    - `decision_tree.pkl` - Trained model
    - `scaler.pkl` - Feature scaler
    - `encoder.pkl` - Label encoders
    """)

with tab2:
    st.markdown("""
    ### Streamlit App Flow
    
    ```
                    [START]
                       |
                       V
              +----------------+
              | User Access    |
              | localhost:8501 |
              +-------+--------+
                      |
                      V
            +------------------+
            | Load Models      |
            | - decision_tree  |
            | - scaler         |
            | - encoder        |
            +--------+---------+
                     |
          +----------+-----------+
          |                      |
          V                      V
    +---------+             +---------+
    | HOME    |             | DATASET |
    |         |             | - Upload| <-- USER
    +---------+             | - View  |
         |                  | - Download|<-- USER
         |                  +---------+
         V                        |
    +---------+                   |
    | EDA     |<------------------+
    |15 Visual|
    +---------+
         |
         V
    +---------+
    | MODEL   |
    |Metrics  |
    +---------+
         |
         V
    +---------+
    | PREDIKSI|<----------- USER INPUT
    | Form    |
    | Result  |
    +---------+
         |
         V
    +---------+
    | TENTANG |
    |  Info   |
    +---------+
    ```
    """)

with tab3:
    st.markdown("""
    ### Stroke Prediction Flow
    
    ```
    [USER INPUT FORM]
         |
         V
    +------------------+
    | Collect Data:    |
    | - Gender         |
    | - Age            |
    | - Hypertension   |
    | - Heart Disease  |
    | - Ever Married   |
    | - Work Type      |
    | - Residence      |
    | - Glucose Level  |
    | - BMI            |
    | - Smoking Status |
    +---------+--------+
              |
              V
    +------------------+
    | Preprocessing    |
    | 1. Encode        |
    | 2. Scale         |
    +---------+--------+
              |
              V
    +------------------+
    | Model Predict    |
    | decision_tree.pkl|
    +---------+--------+
              |
       +------+------+
       |             |
    Stroke=0      Stroke=1
       |             |
       V             V
    +--------+    +--------+
    | SAFE   |    | RISK   |
    |LOW RISK|    |HIGH RISK|
    |  95%   |    |   5%   |
    +--------+    +--------+
       |             |
       +------+------+
              |
              V
    +------------------+
    | Display Result:  |
    | - Probability    |
    | - Visualization  |
    | - Risk Analysis  |
    | - Recommendation |
    +---------+--------+
              |
              V
    +------------------+
    | Download Result  |<-- USER
    | - CSV format     |
    +------------------+
    ```
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Low Risk Output:**
        - Prediction: No Stroke
        - Probability: ~95%
        - Color: Green
        - Message: Reassurance
        - Advice: Maintain health
        """)
    
    with col2:
        st.error("""
        **High Risk Output:**
        - Prediction: Stroke
        - Probability: ~90%+
        - Color: Red
        - Message: Warning
        - Advice: See doctor ASAP
        """)

st.markdown("---")

# =====================================================================
# LICENSE & DISCLAIMER
# =====================================================================
st.markdown("## License & Disclaimer")

col1, col2 = st.columns(2)

with col1:
    st.warning("""
    ### Medical Disclaimer
    
    **IMPORTANT:**
    
    This prediction system is created for **educational and research purposes only**.
    
    - **NOT** a professional medical diagnostic tool
    - **DOES NOT REPLACE** consultation with a doctor
    - Prediction results are for **initial screening only**
    - Accuracy is not 100% guaranteed
    
    **Always consult healthcare professionals for diagnosis and treatment.**
    """)

with col2:
    st.info("""
    ### License
    
    **MIT License**
    
    Copyright (c) 2024
    
    Permission is granted to use, copy, modify, and distribute this software for educational purposes.
    
    **Attribution Required:**
    - Cite this project if used in research
    - Give credit to the team
    - Link to original repository
    
    **No Warranty:**
    - Software provided "as is"
    - No liability for damages
    - Use at your own risk
    """)

st.markdown("---")

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 30px;'>
    <h3>Stroke Prediction System</h3>
    <p><strong>Powered by Decision Tree Algorithm</strong></p>
    <p>Data Mining Project | 2024</p>
    <p>Developed using Python, Streamlit & Machine Learning</p>
    <br>
    <p style='font-size: 0.9rem;'>
        2024 Ferly Ardiansyah, Bayu Aji Yuwono, Wawan suwandi
    </p>
    <p style='font-size: 0.8rem; color: #999;'>
        For educational purposes only | Not for medical diagnosis
    </p>
</div>
""", unsafe_allow_html=True)
