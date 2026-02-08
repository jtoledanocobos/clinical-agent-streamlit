import os
import requests
import streamlit as st

# 1. Configuración de la página
st.set_page_config(page_title="Clinical Readmission Risk Agent", layout="wide")
st.title("🏥 Clinical Readmission Risk Agent")

st.markdown("""
Esta app se conecta a tu endpoint de Model Serving en Databricks
y permite chatear con el agente clínico sobre riesgo de reingreso.
""")

# 2. Leer configuración desde secretos
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
SERVING_ENDPOINT = os.environ.get("DATABRICKS_SERVING_ENDPOINT")

if not (DATABRICKS_HOST and DATABRICKS_TOKEN and SERVING_ENDPOINT):
    st.error("Faltan variables DATABRICKS_HOST, DATABRICKS_TOKEN o DATABRICKS_SERVING_ENDPOINT en los secrets de Streamlit.")
    st.stop()

# 3. Función para llamar al endpoint de Databricks Serving
def call_agent(user_message: str) -> str:
    url = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Formato expected por pyfunc endpoint: pandas dataframe en formato 'dataframe_split'
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
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return f"Error al llamar al endpoint de Databricks: {str(e)}"

    # Respuesta pyfunc: normalmente lista de dicts
    try:
        result = resp.json()
        # Esperamos algo tipo: {"predictions": [{"output": "..."}]} o similar según la config de Serving
        # Si es el formato estándar de pyfunc en Databricks, suele venir en 'predictions'
        predictions = result.get("predictions") or result.get("data") or result
        if isinstance(predictions, list) and len(predictions) > 0:
            first = predictions[0]
            if isinstance(first, dict) and "output" in first:
                return first["output"]
            return str(first)
        else:
            return str(result)
    except Exception as e:
        return f"Error interpretando la respuesta del endpoint: {str(e)}"

# 4. Estado de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Entrada de usuario
user_input = st.chat_input("Describe el caso clínico del paciente...")

if user_input:
    # Guardar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Llamar al agente vía Serving
    with st.chat_message("assistant"):
        with st.spinner("Analizando paciente y protocolos..."):
            output = call_agent(user_input)
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

st.sidebar.markdown("### Ejemplo de prompt")
st.sidebar.code(
    "I have a 75-year-old patient who has been in the hospital for 5 days and has 3 prior visits. "
    "I have diabetes, I don't have hyperglycemia and I have 10 medications. "
    "What is the risk of readmission and what does the discharge protocol say?"
)
