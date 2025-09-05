# app_streamlit.py (VERSÃO CORRIGIDA)

import streamlit as st
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# -------------------------------
# Configurações iniciais
# -------------------------------
st.set_page_config(page_title="Detector de Ervas Daninhas", layout="centered")

# --- CORREÇÃO PRINCIPAL ---
# O caminho para o modelo deve ser relativo à raiz do repositório.
# Linha CORRIGIDA
weights_path = "runs/detect/treinamento_ervas_final/weeds/best.pt"

# Lista de classes do modelo (ajuste se tiver mais de uma)
classes = ["erva daninha"]

# Função para carregar o modelo (com cache para melhor performance)
@st.cache_resource
def load_yolo_model(path):
    """Carrega o modelo YOLO a partir de um caminho."""
    # Verifica se o arquivo do modelo existe antes de carregar
    if not Path(path).exists():
        st.error(f"Arquivo do modelo não encontrado em: {path}")
        st.stop()
    modelo = YOLO(path)
    return modelo

# Carregar modelo treinado
modelo = load_yolo_model(weights_path)

# -------------------------------
# Interface do Streamlit
# -------------------------------
st.title("🌱 Detector de Ervas Daninhas")
st.markdown("Faça upload de uma imagem para verificar se o modelo consegue identificar ervas daninhas.")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lê imagem enviada
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    # Faz predição
    results = modelo.predict(source=img_bgr, conf=0.5, save=False)

    # Mostra resultados
    for r in results:
        # Forçar nomes das classes para exibição nas bounding boxes
        r.names = {0: "erva daninha"} # Garante que a classe 0 sempre seja "erva daninha"
        im_bgr_plot = r.plot()  # imagem com bounding boxes em BGR
        im_rgb_plot = cv2.cvtColor(im_bgr_plot, cv2.COLOR_BGR2RGB) # Converte para RGB para exibição
        
        st.image(im_rgb_plot, caption="Resultado da Detecção", use_column_width=True)

        # Exibir detalhes da predição
        if len(r.boxes) > 0:
            st.subheader("📋 Detalhes da Predição")
            boxes = r.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = classes[cls_id]
                st.write(f"- Objeto {i+1}: Classe **{class_name}**, Confiança: **{conf:.2f}**")
        else:
            st.success("✅ Nenhuma erva daninha detectada na imagem!")
