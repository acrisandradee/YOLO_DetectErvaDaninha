import streamlit as st
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# -------------------------------
# Configurações iniciais
# -------------------------------
st.set_page_config(page_title="Detector de Ervas Daninhas", layout="centered")

project_root = Path.home() / "Documents" / "DeteccaoPlantas"
weights_path = project_root / "runs" / "detect" / "treinamento_ervas_final" / "weights" / "best.pt"

# Lista de classes do modelo
classes = ["erva daninha"]  # índice 0 → "erva daninha"

# Carregar modelo treinado
modelo = YOLO(str(weights_path))

# -------------------------------
# Título
# -------------------------------
st.title("🌱 Detector de Ervas Daninhas")
st.markdown("Faça upload de uma imagem para verificar se o modelo consegue identificar ervas daninhas.")

# -------------------------------
# Upload de imagem
# -------------------------------
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
        r.names = {0: "erva daninha"}
        im_bgr = r.plot()  # imagem com bounding boxes
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        st.image(im_rgb, caption="Resultado da Detecção", use_column_width=True)

        # Exibir detalhes da predição
        st.subheader("📋 Detalhes da Predição")
        boxes = r.boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = classes[cls_id]
            st.write(f"- Objeto {i+1}: Classe {class_name}, Confiança: {conf:.2f}")
