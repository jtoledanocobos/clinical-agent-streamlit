import os
import requests
import streamlit as st

# 1. Page configuration
st.set_page_config(page_title="Clinical Readmission Risk Agent", layout="wide")
st.title("🏥 Clinical Readmission Risk Agent")

st.markdown("""
This app connects to your Databricks Model Serving endpoint
and lets you chat with the clinical agent about readmission risk.
""")

# 2. Read configuration from secrets
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
SERVING_ENDPOINT = os.environ.get("DATABRICKS_SERVING_ENDPOINT")

if not (DATABRICKS_HOST and DATABRICKS_TOKEN and SERVING_ENDPOINT):
    st.error("Missing DATABRICKS_HOST, DATABRICKS_TOKEN, or DATABRICKS_SERVING_ENDPOINT in Streamlit secrets.")
    st.stop()

# 3. Databricks Serving call
def call_databricks_agent(user_message: str) -> str:
    url = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Expected pyfunc payload: pandas dataframe in 'dataframe_split' format
    payload = {
        "dataframe_split": {
            "columns": ["messages"],
            "index": [0],
            "data": [[[
                {"role": "user", "content": user_message}
            ]]]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as e:
        return f"Error calling Databricks endpoint: {str(e)}"

    # Pyfunc response: usually a list of dicts
    try:
        result = response.json()
        # Expected: {"predictions": [{"output": "..."}]} or similar depending on Serving config
        # Standard pyfunc format typically uses 'predictions'
        predictions = result.get("predictions") or result.get("data") or result
        if isinstance(predictions, list) and len(predictions) > 0:
            first_prediction = predictions[0]
            if isinstance(first_prediction, dict) and "output" in first_prediction:
                return first_prediction["output"]
            return str(first_prediction)
        return str(result)
    except Exception as e:
        return f"Error parsing endpoint response: {str(e)}"

# 4. Conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User input
user_input = st.chat_input("Describe the patient's clinical case...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the agent via Serving
    with st.chat_message("assistant"):
        with st.spinner("Analyzing patient and protocols..."):
            output = call_databricks_agent(user_input)
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

st.sidebar.markdown("### Sample prompt")
st.sidebar.code(
    "I have a 75-year-old patient who has been in the hospital for 5 days and has 3 prior visits. "
    "I have diabetes, I don't have hyperglycemia and I have 10 medications. "
    "What is the risk of readmission and what does the discharge protocol say?"
)
