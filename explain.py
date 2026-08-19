import shap
import streamlit as st

@st.cache_resource
def get_shap_explainer(_model, X_train_shape):
    """Creates a SHAP explainer to interpret the XGBoost model predictions. 
    Notice the leading underscore `_model` tells Streamlit not to try to hash the model object."""
    explainer = shap.TreeExplainer(_model)
    return explainer

def generate_shap_explanation(explainer, X_input):
    """Generates SHAP values for a specific prediction input."""
    shap_values = explainer(X_input)
    return shap_values