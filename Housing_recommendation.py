import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import joblib
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
st.set_page_config(layout="wide", page_title="HDB BTO Recommendation System") 
st.title("HDB BTO Price & Affordability Recommender")

# --- 1. Configuration and Setup ---
# --- (Replace with your Google Cloud Project details) ---
PROJECT_ID = "PROJECT-ID" 
LOCATION = "LOCATION"

# --- Constants ---
MODEL_FILENAME = "hdb_price_model.joblib"
BTO_PRICE_DISCOUNT = 0.20  # 20% discount from resale
MORTGAGE_SERVICING_RATIO = 0.30 # 30% MSR
LOAN_TENURE_YEARS = 25
INTEREST_RATE = 0.026 # HDB Concessionary Loan Rate
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# --- Initialize Vertex AI ---
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    gemini_pro_model = GenerativeModel("gemini-2.5-pro")
except Exception as e:
    st.error(f"Failed to initialize Vertex AI. Please check your GCP configuration. Error: {e}")
    gemini_pro_model = None

# --- 2. Data Simulation & Model Training (One-time setup) ---
# In a real scenario, this data would be pulled from BigQuery.
# For this self-contained example, we simulate it.
def load_data_from_csv():
    """Loads HDB resale and BTO launch data from local CSV files."""
    try:
        # <-- CHANGED: Read the new, more detailed CSV file
        resale_df = pd.read_csv("resale_2017.csv") 
        # We'll keep the simple BTO launches file for now
        bto_launches_df = pd.read_csv("hdb_bto_launches.csv")
        st.success("Successfully loaded data from local CSV files.")
        return resale_df, bto_launches_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure 'resale_2017.csv' and 'hdb_bto_launches.csv' are in the same folder as app.py.")
        return None, None

# --- NEW: Data Preprocessing Function ---
def preprocess_data(df: pd.DataFrame):
    """
    Cleans and engineers features from the raw HDB data to make it model-ready.
    """
    st.info("Preprocessing raw data...")

    # --- Feature 1: storey_range_encoded ---
    # Convert text like '01 TO 03' into numerical categories (1: Low, 2: Mid, 3: High)
    def encode_storey(storey_range):
        try:
            # Get the first number in the range (e.g., '04' from '04 TO 06')
            start_floor = int(storey_range.split(' ')[0])
            if start_floor <= 6:
                return 1 # Low floor
            elif start_floor <= 12:
                return 2 # Mid floor
            else:
                return 3 # High floor
        except (ValueError, AttributeError):
            return 2 # Default to mid floor if format is unexpected

    df['storey_range_encoded'] = df['storey_range'].apply(encode_storey)

    # --- Feature 2: remaining_lease_months ---
    # Convert text like '61 years 04 months' into a total number of months
    def convert_lease_to_months(lease_str):
        try:
            years = 0
            months = 0
            parts = lease_str.split(' ')
            if 'years' in parts:
                years_index = parts.index('years')
                years = int(parts[years_index - 1])
            if 'months' in parts:
                months_index = parts.index('months')
                months = int(parts[months_index - 1])
            return (years * 12) + months
        except (ValueError, AttributeError):
            # Fallback using lease_commence_date if remaining_lease is malformed
            # Assuming transaction date is roughly 2024 for this calculation
            try:
                return (99 - (2024 - df['lease_commence_date'])) * 12
            except:
                return 60 * 12 # Default to 60 years if all else fails

    df['remaining_lease_months'] = df['remaining_lease'].apply(convert_lease_to_months)

    st.success("Data preprocessing complete.")
    return df

def train_and_save_model(df: pd.DataFrame):
    """Trains a RandomForest model and saves it locally."""
    st.info("Training a new prediction model...")

    # One-hot encode categorical features
    X_cat = df[['town', 'flat_type']]
    X_num = df[['floor_area_sqm', 'storey_range_encoded', 'remaining_lease_months']]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Save the encoder along with the model
    joblib.dump(encoder, 'one_hot_encoder.joblib')

    X = np.hstack([X_num.values, X_cat_encoded])
    y = df['resale_price']

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the model
    joblib.dump(model, MODEL_FILENAME)
    st.success(f"Model trained and saved as `{MODEL_FILENAME}`.")
    return model, encoder

# --- 3. Core Application Functions ---

def parse_user_prompt_with_llm(user_prompt: str):
    """Uses Vertex AI Gemini to parse user input into structured parameters."""
    if not gemini_pro_model:
        return None

    prompt = f"""
    You are an intelligent assistant for the HDB BTO Recommendation System.
    Your task is to parse the user's request and extract key parameters into a valid JSON object.

    The user's request is: "{user_prompt}"

    Extract the following parameters:
    1.  "bto_launches_max_past_years": The lookback period in years. Default to 10 if not specified.
    2.  "bto_launches_max_count": The maximum number of BTO launches within that period. Default to 1 if not specified.
    3.  "flat_types_to_predict": A list of flat types the user is interested in (e.g., ["3-room", "4-room"]). Default to ["3-room", "4-room"] if not specified.

    Only output the JSON object. Do not include any other text or markdown formatting.

    Example:
    User Request: "Show me towns with almost no BTOs in the last 5 years. I want to see prices for 4 and 5-room flats."
    Your Output:
    {{
        "bto_launches_max_past_years": 5,
        "bto_launches_max_count": 1,
        "flat_types_to_predict": ["4-room", "5-room"]
    }}
    """

    try:
        response = gemini_pro_model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        st.error(f"Error parsing LLM response: {e}. Using default parameters.")
        return {
            "bto_launches_max_past_years": 10,
            "bto_launches_max_count": 1,
            "flat_types_to_predict": ["3-room", "4-room"]
        }

def generate_final_report_with_llm(data: dict):
    """Uses Vertex AI Gemini to generate a natural language report from prediction data."""
    if not gemini_pro_model:
        return "LLM service is unavailable."
    json_data = json.dumps(data, indent=2, cls=NumpyEncoder)
    prompt = f"""
    You are an expert HDB market analyst. Your task is to generate a clear, concise, and helpful report for a user based on the following data.
    The data contains BTO price predictions for towns with limited recent BTO launches.
    
    Data:
    ```json
    {json_data}
    Please structure your response as follows:

    Start with a summary of the recommendations based on the user's criteria.
    For each recommended town, provide a detailed breakdown.
    Explain the predicted BTO prices for each flat type and floor level.
    Clearly state the recommended minimum household income needed for affordability, based on the 30% Mortgage Servicing Ratio (MSR).
    Maintain a professional and helpful tone. """
    response = gemini_pro_model.generate_content(prompt) 
    return response.text

def calculate_affordability(price): 
    """Calculates the recommended monthly income based on MSR."""


    monthly_repayment = npf.pmt(INTEREST_RATE / 12, LOAN_TENURE_YEARS * 12, -price) 
    recommended_income = monthly_repayment / MORTGAGE_SERVICING_RATIO 
    return round(recommended_income, -2) # Round to nearest hundred





resale_df_raw, bto_launches_df = load_data_from_csv()
resale_df = preprocess_data(resale_df_raw)
if not os.path.exists(MODEL_FILENAME): 
    st.warning("Prediction model not found. Training a new one with the loaded data.") 
    train_and_save_model(resale_df)
try: 
    model = joblib.load(MODEL_FILENAME) 
    encoder = joblib.load('one_hot_encoder.joblib') 
except FileNotFoundError: 
    st.error("Could not load model files. Please restart.") 
    st.stop()


st.info("Describe what you're looking for in natural language. The AI will extract the parameters.")

default_prompt = "Please recommend housing estates that have had limited BTO launches in the past ten years. For each estate, provide an analysis of potential BTO prices for both 3-room and 4-room flats." 

user_input = st.text_area("Your Request:", value=default_prompt, height=100)

if st.button("Generate Recommendation"): 
    if not user_input: 
        st.warning("Please enter a request.") 
    else: 
        with st.spinner("AI is analyzing your request and generating recommendations..."): 
            # 1. Parse user prompt with LLM st.subheader("1. AI Parameter Extraction") 
            params = parse_user_prompt_with_llm(user_input) 
            st.json(params)

        # 2. Find eligible towns based on BTO history
        current_year = 2025
        lookback_year = current_year - params['bto_launches_max_past_years']

        recent_launches = bto_launches_df[bto_launches_df['launch_year'] >= lookback_year]
        launch_counts = recent_launches.groupby('town').size()

        all_towns = resale_df['town'].unique()
        eligible_towns = [
            town for town in all_towns 
            if launch_counts.get(town, 0) <= params['bto_launches_max_count']
            ]

        # 3. Predict prices for archetypes in eligible towns
        results = {"recommendations": []}
        flat_type_map = {
            "3-room": {"area": 68, "type_str": "3 ROOM"},
            "4-room": {"area": 93, "type_str": "4 ROOM"},
            "5-room": {"area": 112, "type_str": "5 ROOM"}
        }

        for town in eligible_towns:
            town_result = {
                "town": town,
                "bto_launches_in_period": launch_counts.get(town, 0),
                "price_predictions": []
            }
            for flat_type_key in params['flat_types_to_predict']:
                if flat_type_key in flat_type_map:
                    flat_info = flat_type_map[flat_type_key]
                    for level, storey_code in [("low", 1), ("middle", 2), ("high", 3)]:
                        # Create a feature vector for prediction
                        archetype_df = pd.DataFrame({
                            'town': [town],
                            'flat_type': [flat_info['type_str']],
                            'floor_area_sqm': [flat_info['area']],
                            'storey_range_encoded': [storey_code],
                            'remaining_lease_months': [1140] # New BTO: 95 years * 12
                        })

                        # Encode categorical features
                        cat_encoded = encoder.transform(archetype_df[['town', 'flat_type']])
                        num_features = archetype_df[['floor_area_sqm', 'storey_range_encoded', 'remaining_lease_months']].values

                        features = np.hstack([num_features, cat_encoded])

                        # Predict resale price
                        predicted_resale = model.predict(features)[0]
                        estimated_bto = predicted_resale * (1 - BTO_PRICE_DISCOUNT)
                        income_needed = calculate_affordability(estimated_bto)

                        town_result["price_predictions"].append({
                            "flat_type": flat_type_key,
                            "floor_level": level,
                            "predicted_resale_price": int(predicted_resale),
                            "estimated_bto_price": int(estimated_bto),
                            "recommended_household_income": int(income_needed)
                        })
            results["recommendations"].append(town_result)

            # 4. Generate final report with LLM
            st.subheader("2. Final Report")
            final_report = generate_final_report_with_llm(results)
            st.markdown(final_report)

            # 5. Show raw data for transparency
            with st.expander("View Raw Prediction Data"):
                st.json(results)

