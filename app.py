import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Import your custom modules
import dummy_coding
import imp_norm
import cox_model
import deepsurv_model
import deephit_model
import xgboost_model

# Import required classes for model loading
from torchtuple_cox import _CoxPHBase
from pycox.models import loss as pycox_loss
from torchtuple_pmf import PMFBase
from pycox import models


# Define custom classes (same as Flask version)
class CoxPH_new(_CoxPHBase):
    def __init__(self, net, optimizer=None, device=None, loss=None, scheduler=None):
        if loss is None:
            loss = pycox_loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)
        self.scheduler = scheduler


class DeepHitSingle_new(PMFBase):
    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1, loss=None, scheduler=None):
        if loss is None:
            loss = pycox_loss.DeepHitSingleLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)
        self.scheduler = scheduler

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader


# Streamlit App Configuration
st.set_page_config(
    page_title="Esophagectomy Survival Calculator",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Recurrence & Survival Calculator")
st.subheader("For Patients after Curative-intent Esophagectomy")
st.markdown("---")

# Create form
with st.form("calculator_form"):
    # Organize inputs in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sex = st.selectbox("Sex", ["Select", "Female", "Male", "NA"], key="sex")
        age = st.text_input("Age at Diagnosis", key="age")
        asa_grade = st.selectbox("ASA Grade", ["Select", "1", "2", "3", "NA"], key="asa")
        barretts = st.selectbox("Barrett's Esophagus", ["Select", "No", "Yes", "NA"], key="barretts")
        histologic = st.selectbox("Histologic Type", ["Select", "Adenocarcinoma", "Squamous Cell Carcinoma", "NA"], key="histologic")
    
    with col2:
        clinical_t = st.selectbox("Clinical T Stage", ["Select", "0", "1", "2", "3", "4a", "4b", "X", "NA"], key="clinical_t")
        clinical_n = st.selectbox("Clinical N Stage", ["Select", "0", "1", "2", "3", "X", "NA"], key="clinical_n")
        clinical_diff = st.selectbox("Clinical Differentiation", ["Select", "Well", "Moderate", "Poor", "Undifferentiated", "NA"], key="clinical_diff")
        tumor_site = st.selectbox("Tumour Site", ["Select", "Upper third", "Middle third", "Lower third", "GOJ", "NA"], key="tumor_site")
        path_t = st.selectbox("Pathologic T Stage", ["Select", "0", "1a", "1b", "2", "3", "4a", "4b", "X", "NA"], key="path_t")
    
    with col3:
        path_n = st.selectbox("Pathologic N Stage", ["Select", "0", "1", "2", "3", "X", "NA"], key="path_n")
        differentiation = st.selectbox("Differentiation", ["Select", "Well", "Moderate", "Poor", "Undifferentiated", "NA"], key="differentiation")
        num_nodes = st.text_input("Number of Nodes Analyzed", key="num_nodes")
        treatment = st.selectbox("Treatment Protocol", ["Select", "Surgery", "NAC then Surgery", "Surgery then AC", "Surgery then NACTRT", "NACTRT then Surgery", "NA"], key="treatment")
        operation = st.selectbox("Operation Type", ["Select", "Oesophagectomy", "Oesophagogastrectomy", "NA"], key="operation")
    
    with col4:
        approach = st.selectbox("Approach", ["Select", "Open", "Hybrid-MIE", "Total-MIE", "NA"], key="approach")
        r_status = st.selectbox("R status", ["Select", "0", "1", "NA"], key="r_status")
        los = st.text_input("Length of Stay", key="los")
        recurrent_nerve = st.selectbox("Recurrent Laryngeal Nerve Injury", ["Select", "0", "1", "NA"], key="recurrent_nerve")
        chyle_leak = st.selectbox("Chyle Leak", ["Select", "0", "1", "NA"], key="chyle_leak")
    
    with col5:
        leak = st.selectbox("Gastroenteric Leak", ["Select", "0", "1", "NA"], key="leak")
        gdp = st.text_input("GDP USD per cap", key="gdp")
        h_voluntary = st.text_input("H voluntary", key="h_voluntary")
        he_compulsory = st.text_input("HE Compulsory", key="he_compulsory")
        cancer_cases = st.selectbox("Cancer Cases per Year", ["Select", "0-10", "10-20", "20-30", "30-40", "40-50", "50-60", ">60", "NA"], key="cancer_cases")
    
    # Additional inputs in a new row
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        consultant_volume = st.selectbox("Consultant Volume SPSS", ["Select", "0", "1", "2", "NA"], key="consultant_volume")
    
    with col7:
        intensive_surv = st.selectbox("Intensive Surveillance", ["Select", "0", "1", "NA"], key="intensive_surv")
    
    with col8:
        clinical_endpoint = st.selectbox("Clinical Endpoint", ["Select", "DFS", "OS", "DSS"], key="clinical_endpoint")
    
    with col9:
        survival_model = st.selectbox("Survival Model", ["Select", "CoxPH", "XGBoost", "DeepSurv", "DeepHit"], key="survival_model")
    
    # Submit button
    submitted = st.form_submit_button("Calculate Survival", use_container_width=True)

# Process form submission
if submitted:
    # Validate inputs
    if sex == "Select" or clinical_endpoint == "Select" or survival_model == "Select":
        st.error("Please select all required fields!")
    else:
        try:
            # Map values to match your original Flask format
            sex_map = {"Female": "0", "Male": "1", "NA": "NA"}
            barretts_map = {"No": "0", "Yes": "1", "NA": "NA"}
            histologic_map = {"Adenocarcinoma": "0", "Squamous Cell Carcinoma": "1", "NA": "NA"}
            
            # Map cancer cases
            cancer_map = {"0-10": "0", "10-20": "1", "20-30": "2", "30-40": "3", "40-50": "4", "50-60": "5", ">60": "6", "NA": "NA"}
            
            # Create data dictionary
            data = {
                "Sex": sex_map.get(sex, sex),
                "Age at diagnosis": age,
                "ASA grade": asa_grade if asa_grade != "Select" else "NA",
                "Barrett's esophagus": barretts_map.get(barretts, barretts),
                "Histologic type": histologic_map.get(histologic, histologic),
                "Clinical T stage": clinical_t if clinical_t != "Select" else "NA",
                "Clinical N stage": clinical_n if clinical_n != "Select" else "NA",
                "Clinical Differentiation": clinical_diff if clinical_diff != "Select" else "NA",
                "Tumour site": tumor_site if tumor_site != "Select" else "NA",
                "Pathologic T stage": path_t if path_t != "Select" else "NA",
                "Pathologic N stage": path_n if path_n != "Select" else "NA",
                "Differentiation": differentiation if differentiation != "Select" else "NA",
                "Number of nodes analyzed": num_nodes,
                "Treatment protocol": treatment if treatment != "Select" else "NA",
                "Operation type": operation if operation != "Select" else "NA",
                "Approach": approach if approach != "Select" else "NA",
                "R status": r_status if r_status != "Select" else "NA",
                "Length of stay": los,
                "Recurrent laryngeal nerve injury": recurrent_nerve if recurrent_nerve != "Select" else "NA",
                "Chyle leak": chyle_leak if chyle_leak != "Select" else "NA",
                "Gastroenteric leak": leak if leak != "Select" else "NA",
                "GDP USD per cap": gdp,
                "H volutary": h_voluntary,
                "HE Compulsory": he_compulsory,
                "Cancer cases per year": cancer_map.get(cancer_cases, cancer_cases),
                "Consultant Volume SPSS": consultant_volume if consultant_volume != "Select" else "NA",
                "Intensive Surveillance": intensive_surv if intensive_surv != "Select" else "NA"
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Replace 'NA' with np.nan
            df.replace('NA', np.nan, inplace=True)
            
            # Convert numeric columns to float
            df = df.map(lambda x: float(x) if pd.notnull(x) else x)
            
            # Show processing message
            with st.spinner('Processing data and generating survival plot...'):
                # Dummy code the data
                df_dc = dummy_coding.dummy_code_data(df)
                
                # Impute and normalize
                df_imp_norm = imp_norm.imp_norm(df_dc, clinical_endpoint)
                
                # Generate prediction based on model
                if survival_model == 'CoxPH':
                    encoded_image = cox_model.cox_model(df_imp_norm, clinical_endpoint)
                elif survival_model == 'DeepSurv':
                    encoded_image = deepsurv_model.deepsurv_model(df_imp_norm, clinical_endpoint)
                elif survival_model == 'DeepHit':
                    encoded_image = deephit_model.deephit_model(df_imp_norm, clinical_endpoint)
                elif survival_model == 'XGBoost':
                    encoded_image = xgboost_model.xgboost_model(df, clinical_endpoint)
            
            # Display results
            st.success("✅ Prediction Complete!")
            st.markdown("---")
            st.subheader("📊 Survival Plot")
            
            # Display the image
            import base64
            from io import BytesIO
            image_data = base64.b64decode(encoded_image)
            st.image(image_data, use_container_width=True)
            
            # Add download button
            st.download_button(
                label="Download Plot",
                data=image_data,
                file_name=f"survival_plot_{survival_model}_{clinical_endpoint}.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your inputs and try again.")