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
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Center the titles and make them smaller */
    h1 {
        text-align: center;
        font-size: 2rem !important;  /* Smaller title (default is ~2.5rem) */
        margin-top: -50px !important;  /* Move up */
        margin-bottom: 5px !important;  /* Less space below */
    }
    h3 {
        text-align: center;
        font-size: 1.4rem !important;  /* Smaller subtitle */
        margin-top: 0px !important;
        margin-bottom: 10px !important;
    }

    /* =========================
    Calculate (Submit) button
    ========================= */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #4CAF50 !important;   /* green */
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #45a049 !important;   /* darker green */
        color: white !important;
    }

    /* =========================
    Reset button
    ========================= */
    div[data-testid="stButton"] > button {
        background-color: #E53935 !important;   /* red */
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #C62828 !important;   /* darker red */
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Title
st.title("Recurrence & Survival Calculator")
st.subheader("For Patients after Curative-intent Esophagectomy")


def reset_form():
    defaults = {
        "sex": "Select",
        "clinical_t": "Select",
        "distal_margin": "Select",
        "path_m": "Select",
        "num_nodes": "",
        "clavien_dindo": "Select",
        "he_compulsory": "",
        "age": "",
        "clinical_n": "Select",
        "proximal_margin": "Select",
        "diff": "Select",
        "treatment": "Select",
        "los": "",
        "cancer_cases": "Select",
        "asa": "Select",
        "clinical_m": "Select",
        "radial_margin": "Select",
        "lymphatic_inva": "Select",
        "operation": "Select",
        "gastro_leak": "Select",
        "consultant_volume": "Select",
        "barretts": "Select",
        "clinical_diff": "Select",
        "path_t": "Select",
        "venous_inva": "Select",
        "approach": "Select",
        "gdp": "",
        "intensive_surv": "Select",
        "histologic": "Select",
        "tumor_site": "Select",
        "path_n": "Select",
        "peri_invasion": "Select",
        "robotic": "Select",
        "h_volutary": "",
        "clinical_endpoint": "Select",
        "survival_model": "Select",
    }


    st.session_state.update(defaults)
    st.session_state.encoded_image = None


col1, col2, col3 = st.columns([5, 1, 1])
with col3:
    st.button("Reset", use_container_width=True, on_click=reset_form)

# Create form 
with st.form(f"calculator_form"):
    # Organize inputs in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sex = st.selectbox("Sex", ["Select", "Female", "Male", "NA"], key="sex")
        clinical_t = st.selectbox("Clinical T Stage", ["Select", "cTis/HGD", "cT1", "cT2", "cT3", "cT4", "NA"], key="clinical_t")
        distal_margin = st.selectbox("Distal margin positive", ["Select", "No", "Yes", "NA"], key="distal_margin")
        path_m = st.selectbox("Pathologic M Stage", ["Select", "M0", "M1", "NA"], key="path_m")
        num_nodes = st.text_input("Number of nodes analyzed", placeholder="Enter number or NA", key="num_nodes")
        clavien_dindo = st.selectbox("Clavien-Dindo Grade", ["Select", "No complications", "Grade I", "Grade II", "Grade IIIa", "Grade IIIb", "Grade IVa", "Grade IVb", "Grade V (mortality)", "NA"], key="clavien_dindo")
        he_compulsory = st.text_input("HE Compulsory", placeholder="Enter number or NA", key="he_compulsory")
       
    
    with col2:
        age = st.text_input("Age at Diagnosis", placeholder="Enter number or NA", key="age")
        clinical_n = st.selectbox("Clinical N stage", ["Select", "cN0", "cN1", "cN2", "cN3", "NA"], key="clinical_n")
        proximal_margin = st.selectbox("Proximal margin positive", ["Select", "No", "Yes", "NA"], key="proximal_margin")
        diff = st.selectbox("Differentiation", ["Select", "Gx, cannot be assessed", "Well differentiated", "Moderately differentiated", "Poorly differentiated", "Signet ring features", "NA"], key="diff")
        treatment = st.selectbox("Treatment protocol", ["Select", "Surgery only", 
                                                        "Surgery and adjuvant chemotherapy and/or radiation", 
                                                        "Neoadjuvant chemotherapy then surgery (±perioperative)", 
                                                        "Neoadjuvant chemoradiation then surgery", 
                                                        "Definitive chemoradiation and salvage surgery", 
                                                        "NA"], key="treatment")
        los = st.text_input("Length of stay", placeholder="Enter number or NA", key="los")
        cancer_cases = st.selectbox("Cancer cases per year", ["Select", "0-10", "10-20", "20-30", "30-40", "40-50", "50-60", ">60", "NA"], key="cancer_cases")


    with col3:
        asa = st.selectbox("ASA Grade", ["Select", "1", "2", "3", "NA"], key="asa")
        clinical_m = st.selectbox("Clinical M Stage", ["Select", "M0", "M1 distant metastases", "NA"], key="clinical_m")
        radial_margin = st.selectbox("Radial margin positive", ["Select", "No", "Yes", "NA"], key="radial_margin")
        lymphatic_inva = st.selectbox("Lymphatic invasion", ["Select", "Not present", "Present", "NA"], key="lymphatic_inva")
        operation = st.selectbox("Operation type", ["Select", "Extended total gastrectomy Roux-en-Y", 
                                                    "2-stage esophagecomy (Ivor Lewis)", 
                                                    "3-stage esophagectomy (McKeown)", 
                                                    "Transhiatal esophagectomy", 
                                                    "Pharyngo-largyngo-esophagectomy", 
                                                    "Sweet (left thoracoabdominal) + cervical anastomosis", 
                                                    "Sweet (left thoracoabdominal) + intrathoracic anastomosis", 
                                                    "NA"], key="operation")
        gastro_leak = st.selectbox("Gastroenteric leak", ["Select", "No Leak", "Leak", "NA"], key="gastro_leak")
        consultant_volume = st.selectbox("Consultant Volume SPSS", ["Select", "0", "1", "2", "NA"], key="consultant_volume")

    
    with col4:
        barretts = st.selectbox("Barrett's Esophagus", ["Select", "No", "Yes", "NA"], key="barretts")
        clinical_diff = st.selectbox("Clinical Differentiation", ["Select", "Gx, cannot be assessed", "Well differentiated", "Moderately differentiated", "Poorly differentiated", "Signet ring", "NA"], key="clinical_diff")
        path_t = st.selectbox("Pathologic T stage", ["Select", "T0", "Tis/HGD", "T1", "T2", "T3", "T4", "NA"], key="path_t")
        venous_inva = st.selectbox("Venous invasion", ["Select", "Not present", "Present", "NA"], key="venous_inva")
        approach = st.selectbox("Approach", ["Select", "Open", "Hybrid MIE", "Total MIE", "Hybrid converted", "Total MIE converted", "NA"], key="approach")
        gdp = st.text_input("GDP USD per cap", placeholder="Enter number or NA", key="gdp")
        intensive_surv = st.selectbox("Intensive Surveillance", ["Select", "No", "Yes", "NA"], key="intensive_surv")


    with col5:
        histologic = st.selectbox("Histologic Type", ["Select", "Adenocarcinoma", "Squamous Cell Carcinoma", "NA"], key="histologic")
        tumor_site = st.selectbox("Tumour site", ["Select", "Junctional", "Lower", "Middle", "Upper", "NA"], key="tumor_site")
        path_n = st.selectbox("Pathologic N stage", ["Select", "N0", "N1", "N2", "N3", "NA"], key="path_n")
        peri_invasion = st.selectbox("Perineural invasion", ["Select", "Not present", "Present", "NA"], key="peri_invasion")
        robotic = st.selectbox("Robotic assisted", ["Select", "No", "Yes, robotic assisted", "NA"], key="robotic")
        h_volutary = st.text_input("H volutary", placeholder="Enter number or NA", key="h_volutary")

  
    # Additional inputs in a new row
    st.markdown("---")
    col6, col7 = st.columns(2)

    with col6:
        st.markdown('<p style="color: #0D4A70; font-weight: bold;">📊 Analysis Options</p>', unsafe_allow_html=True)
        clinical_endpoint = st.selectbox("Clinical Endpoint", ["Select", "DFS", "OS", "DSS"], key="clinical_endpoint")

    with col7:
        st.markdown('<p style="color: #0D4A70; font-weight: bold;">🤖 Model Selection</p>', unsafe_allow_html=True)
        survival_model = st.selectbox("Survival Model", ["Select", "CoxPH", "XGBoost", "DeepSurv", "DeepHit"], key="survival_model")
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Calculate", use_container_width=True, type="primary")

# Process form submission
if submitted:
    # Validate inputs
    if clinical_endpoint == "Select" or survival_model == "Select":
        st.error("Please select a Clinical Endpoint and a Survival Model")
    else:
        try:
            # Map values to match your original Flask format
            sex_map = {"Select": "NA", "Female": "0", "Male": "1", "NA": "NA"}
            clinical_t_map = {"Select": "NA", "cTis/HGD": "0", "cT1": "1", "cT2": "2", "cT3": "3", "cT4": "4", "NA": "NA"}
            distal_margin_map = {"Select": "NA", "No": "0", "Yes": "1", "NA": "NA"}
            path_m_map = {"Select": "NA", "M0": "0", "M1": "1", "NA": "NA"}
            clavien_dindo_map = {"Select": "NA", "No complications": "0", "Grade I": "1", "Grade II": "2", "Grade IIIa": "3", "Grade IIIb": "4", "Grade IVa": "5", "Grade IVb": "6", "Grade V (mortality)": "7", "NA": "NA"}


            clinical_n_map = {"Select": "NA", "cN0": "0", "cN1": "1", "cN2": "2", "cN3": "3", "NA": "NA"}
            proximal_margin_map = {"Select": "NA", "No": "0", "Yes": "1", "NA": "NA"}
            diff_map = {"Select": "NA", "Gx, cannot be assessed": "0", "Well differentiated": "1", "Moderately differentiated": "2", "Poorly differentiated": "3", "Signet ring features": "4", "NA": "NA"}
            treatment_map = {"Select": "NA", "Surgery only": "0", 
                             "Surgery and adjuvant chemotherapy and/or radiation": "1", 
                             "Neoadjuvant chemotherapy then surgery (±perioperative)": "2", 
                             "Neoadjuvant chemoradiation then surgery": "3", 
                             "Definitive chemoradiation and salvage surgery": "4", 
                             "NA": "NA"}
            cancer_map = {"Select": "NA", "0-10": "0", "10-20": "1", "20-30": "2", "30-40": "3", "40-50": "4", "50-60": "5", ">60": "6", "NA": "NA"}


            clinical_m_map = {"Select": "NA", "M0": "0", "M1 distant metastases": "1", "NA": "NA"}  
            radial_margin_map = {"Select": "NA", "No": "0", "Yes": "1", "NA": "NA"}
            lymphatic_inva_map = {"Select": "NA", "Not present": "0", "Present": "1", "NA": "NA"}
            operation_map = {"Select": "NA", "Extended total gastrectomy Roux-en-Y": "0", 
                             "2-stage esophagecomy (Ivor Lewis)": "1", 
                             "3-stage esophagectomy (McKeown)": "2",
                             "Transhiatal esophagectomy": "3",
                             "Pharyngo-largyngo-esophagectomy": "4",
                             "Sweet (left thoracoabdominal) + cervical anastomosis": "5",
                             "Sweet (left thoracoabdominal) + intrathoracic anastomosis": "6",
                             "NA": "NA"}
            gastro_leak_map = {"Select": "NA", "No Leak": "0", "Leak": "1", "NA": "NA"}


            barretts_map = {"Select": "NA", "No": "0", "Yes": "1", "NA": "NA"}
            clinical_diff_map = {"Select": "NA", "Gx, cannot be assessed": "0", "Well differentiated": "1", "Moderately differentiated": "2", "Poorly differentiated": "3", "Signet ring": "4", "NA": "NA"}
            path_t_map = {"Select": "NA", "T0": "0", "Tis/HGD": "1", "T1": "2", "T2": "3", "T3": "4", "T4": "5", "NA": "NA"}
            venous_inva_map = {"Select": "NA", "Not present": "0", "Present": "1", "NA": "NA"}
            approach_map = {"Select": "NA", "Open": "0", "Hybrid MIE": "1", "Total MIE": "2", "Hybrid converted": "3", "Total MIE converted": "4", "NA": "NA"}
            intense_surv_map = {"Select": "NA", "No": "0", "Yes": "1", "NA": "NA"}


            histologic_map = {"Select": "NA", "Adenocarcinoma": "0", "Squamous Cell Carcinoma": "1", "NA": "NA"}
            tumor_site_map = {"Select": "NA", "Junctional": "0", "Lower": "1", "Middle": "2", "Upper": "3", "NA": "NA"}
            path_n_map = {"Select": "NA", "N0": "0", "N1": "1", "N2": "2", "N3": "3", "NA": "NA"}
            peri_invasion_map = {"Select": "NA", "Not present": "0", "Present": "1", "NA": "NA"}
            robotic_map = {"Select": "NA", "No": "0", "Yes, robotic assisted": "1", "NA": "NA"}



            # Create data dictionary
            data = {
                "Sex": sex_map.get(sex, sex), # allow for NA?
                "Age at diagnosis": age if age else "NA",
                "ASA grade": asa if asa != "Select" else "NA",
                "Barrett's esophagus": barretts_map.get(barretts, barretts),
                "Histologic type": histologic_map.get(histologic, histologic),


                "Clinical T stage": clinical_t_map.get(clinical_t, clinical_t),
                "Clinical N stage": clinical_n_map.get(clinical_n, clinical_n),
                "Clinical M stage": clinical_m_map.get(clinical_m, clinical_m),
                "Clinical Differentiation": clinical_diff_map.get(clinical_diff, clinical_diff),
                "Tumour site": tumor_site_map.get(tumor_site, tumor_site),


                "Distal margin positive": distal_margin_map.get(distal_margin, distal_margin),
                "Proximal margin positive": proximal_margin_map.get(proximal_margin, proximal_margin),
                "Radial margin positive": radial_margin_map.get(radial_margin, radial_margin),
                "Pathologic T stage": path_t_map.get(path_t, path_t),
                "Pathologic N stage": path_n_map.get(path_n, path_n),


                "Pathologic M Stage": path_m_map.get(path_m, path_m),
                "Differentiation": diff_map.get(diff, diff),
                "Lymphatic invasion": lymphatic_inva_map.get(lymphatic_inva, lymphatic_inva),
                "Venous invasion": venous_inva_map.get(venous_inva, venous_inva),
                "Perineural invasion": peri_invasion_map.get(peri_invasion, peri_invasion),


                "Number of nodes analyzed": num_nodes if num_nodes else "NA",
                "Treatment protocol": treatment_map.get(treatment, treatment),
                "Operation type": operation_map.get(operation, operation),
                "Approach": approach_map.get(approach, approach),
                "Robotic assisted": robotic_map.get(robotic, robotic),


                "Clavien-Dindo Grade": clavien_dindo_map.get(clavien_dindo, clavien_dindo),
                "Length of stay": los if los else "NA",
                "Gastroenteric leak": gastro_leak_map.get(gastro_leak, gastro_leak),
                "GDP USD per cap": gdp if gdp else "NA",
                "H volutary": h_volutary if h_volutary else "NA",



                "HE Compulsory": he_compulsory if he_compulsory else "NA",
                "Cancer cases per year": cancer_map.get(cancer_cases, cancer_cases),
                "Consultant Volume SPSS": consultant_volume if consultant_volume != "Select" else "NA",
                "Intensive Surveillance": intense_surv_map.get(intensive_surv, intensive_surv),

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
            st.subheader("📊 Prediction Result")
            
            # Display the image with smaller width (centered)
            import base64
            from io import BytesIO
            image_data = base64.b64decode(encoded_image)
            
            # Use columns to make image smaller and centered
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_data, use_container_width=True)
            
            # Center the download button too
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Save Prediction Result",
                    data=image_data,
                    file_name=f"prediction_result_{survival_model}_{clinical_endpoint}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your inputs and try again.")
