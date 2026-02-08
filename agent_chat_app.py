import streamlit as st
import mlflow

# 1. Configuración de la página
st.set_page_config(page_title="Clinical Readmission Risk Agent", layout="wide")
st.title("🏥 Clinical Readmission Risk Agent")

st.markdown("""
Esta app se conecta a tu modelo registrado en Databricks (MLflow pyfunc)
y permite chatear con el agente clínico sobre riesgo de reingreso.
""")

# 2. Cargar el modelo una sola vez
@st.cache_resource
def load_agent():
    model_name = "emea_databricks_hackathon_2025.emea_emea_south_atc_databricks_reinventors_hackathon.clinical_agent"
    return mlflow.pyfunc.load_model(f"models:/{model_name}@production")

agent = load_agent()

# 3. Estado de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Mostrar el historial de chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Entrada del usuario
user_input = st.chat_input("Describe el caso clínico del paciente...")

if user_input:
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Llamar al agente
    with st.chat_message("assistant"):
        with st.spinner("Analizando paciente y protocolos..."):
            try:
                payload = [{
                    "messages": [
                        {"role": "user", "content": user_input}
                    ]
                }]
                response = agent.predict(payload)
                output = response[0]["output"]
            except Exception as e:
                output = f"Error al llamar al agente: {str(e)}"

            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

st.sidebar.markdown("### Ejemplo de prompt")
st.sidebar.code(
    "I have a 75-year-old patient who has been in the hospital for 5 days and has 3 prior visits. "
    "I have diabetes, I don't have hyperglycemia and I have 10 medications. "
    "What is the risk of readmission and what does the discharge protocol say?"
)
