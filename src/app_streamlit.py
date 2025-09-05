import streamlit as st
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Ervas Daninhas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARREGAMENTO DO MODELO ---
weights_path = "runs/detect/treinamento_ervas_final/weights/best.pt"
classes = ["erva daninha"]

@st.cache_resource
def load_yolo_model(path):
    if not Path(path).exists():
        st.error(f"Arquivo do modelo não encontrado em: {path}")
        st.stop()
    modelo = YOLO(path)
    return modelo

modelo = load_yolo_model(weights_path)


st.sidebar.title("Painel de Controle ")
st.sidebar.markdown("Ajuste os parâmetros de detecção.")

confidence_threshold = st.sidebar.slider(
    "Nível de Confiança da Detecção", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,  
    step=0.05
)

st.title("🌿 Detector Inteligente de Ervas Daninhas")
st.markdown(
    "Faça o upload de uma imagem do seu jardim ou plantação e nossa IA fará a detecção de ervas daninhas."
)


uploaded_file = st.file_uploader(
    "Selecione uma imagem para análise", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(img_rgb, use_container_width=True)

    with st.spinner("Analisando a imagem... "):
        results = modelo.predict(source=img_bgr, conf=confidence_threshold, save=False)
        r = results[0] 

        im_bgr_plot = r.plot()
        im_rgb_plot = cv2.cvtColor(im_bgr_plot, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Resultado da Detecção")
        st.image(im_rgb_plot, use_container_width=True)

    if len(r.boxes) > 0:
        with st.expander("Clique para ver os detalhes da predição 👇"):
            boxes = r.boxes
            st.write(f"**Total de detecções:** {len(boxes)}")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = classes[cls_id]
                st.write(f"- **Objeto {i+1}:** Classe `{class_name}`, Confiança: `{conf:.2f}`")
    else:
        st.success("✅ Nenhuma erva daninha detectada com o nível de confiança atual!")

else:
    st.info("Aguardando o upload de uma imagem.")
    

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: grey;">
        <p>🌱 Software desenvolvido por <strong>Cristina Andrade</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)