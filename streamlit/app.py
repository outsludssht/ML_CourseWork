import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Draw
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.ML.Descriptors import MoleculeDescriptors

@st.cache_resource
def load_assets():
    model = joblib.load('final_model.pkl')
    svd = joblib.load('svd_transformer.pkl')
    feature_names = joblib.load('feature_names.pkl') 
    explainer = shap.Explainer(model)
    return model, svd, feature_names, explainer

model, svd, feature_names, explainer = load_assets()

def process_features(smi, weight, alogp, svd_model, final_cols):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None

    # 1. RDKit Descriptors
    desc_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rdkit_feats = list(calculator.CalcDescriptors(mol))
    df_rdkit = pd.DataFrame([rdkit_feats], columns=desc_names)

    # 2. SVD Fingerprints
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((1024,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    fp_reduced = svd_model.transform(arr.reshape(1, -1))
    df_svd = pd.DataFrame(fp_reduced, columns=[f'svd_fp_{i}' for i in range(50)])

    # 3. Custom features
    full_df = pd.concat([df_rdkit, df_svd], axis=1)
    full_df['Log_MolWt'] = np.log1p(weight)
    full_df['LogP_Wt_Ratio'] = alogp / (weight + 1e-6)

    # 4. Alignment
    for col in final_cols:
        if col not in full_df.columns:
            full_df[col] = 0
            
    return full_df[final_cols]

# --- Streamlit UI ---
st.set_page_config(page_title="BioActivity Predictor", layout="wide")
st.title("🧪 Activity Predictor of MAO-B inhibitors")

with st.sidebar:
    st.header("Input Parameters")
    user_smi = st.text_input("SMILES string:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    
    mol_tmp = Chem.MolFromSmiles(user_smi)
    if mol_tmp:
        default_w = Descriptors.MolWt(mol_tmp)
        default_p = Descriptors.MolLogP(mol_tmp)
        
        user_weight = st.number_input("Molecular Weight:", value=float(default_w))
        user_alogp = st.number_input("AlogP:", value=float(default_p))
    else:
        st.error("Invalid SMILES")

col_mol, col_pred = st.columns([1, 1])

if mol_tmp:
    with col_mol:
        st.subheader("Molecule Structure")
        img = Draw.MolToImage(mol_tmp, size=(400, 400))
        st.image(img, use_container_width=True)

if st.button("🚀 Calculate activity & Explain"):
    if mol_tmp:
        with st.spinner('Calculating SHAP values...'):
            input_df = process_features(user_smi, user_weight, user_alogp, svd, feature_names)

            prediction = model.predict(input_df)[0]
            
            with col_pred:
                st.subheader("Result")
                st.metric(label="Predicted pChEMBL Value", value=f"{prediction:.4f}")

            # --- SHAP Visualization ---
            st.divider()
            st.subheader("Feature Importance (Local Interpretation)")

            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Please enter a valid SMILES.")