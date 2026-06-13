import cv2
from ultralytics import YOLO

# 1. Carregar o modelo pré-treinado
# O sufixo 'n' significa Nano (mais rápido, menos preciso). 
# Tente 'yolov8s.pt' ou 'yolov8m.pt' se tiver uma GPU potente e quiser mais precisão.
model = YOLO('yolo11n.pt') 

# 2. Definir a fonte de vídeo
# Use '0' para sua Webcam, ou o caminho para um arquivo de vídeo 'transito.mp4'
source_path = "video.mp4" # Troque por 0 para usar a webcam
cap = cv2.VideoCapture(source_path)

# IDs das classes de veículos no dataset COCO (usado no treinamento padrão do YOLO)
# 1: bicicleta, 2: carro, 3: moto, 5: ônibus, 7: caminhão
vehicle_classes = [1, 2, 3, 5, 7]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Fim do vídeo ou erro na câmera.")
        break

    frame = cv2.resize(frame, (1280, 720))

    # 3. Realizar a detecção
    # classes=vehicle_classes garante que só detectamos veículos
    # conf=0.5 exige 50% de certeza para mostrar o objeto
    results = model.predict(frame, conf=0.5, classes=vehicle_classes, verbose=False)

    # 4. Visualizar os resultados
    # O método .plot() desenha as caixas delimitadoras na imagem automaticamente
    annotated_frame = results[0].plot()

    # Mostrar na tela
    cv2.imshow("Detecção de Veículos - YOLO", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()