import cv2
import torch
from ultralytics import YOLO

# Verificação visual se a GPU está disponível
print(f"CUDA disponível? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("AVISO: Rodando em CPU. Verifique a instalação do PyTorch.")

model = YOLO('yolo11n.pt') 
source_path = "video.mp4" 
cap = cv2.VideoCapture(source_path)
vehicle_classes = [1, 2, 3, 5, 7]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1280, 720))

    # AQUI ESTÁ A MUDANÇA: device=0
    # Se você não tiver GPU, mude para device='cpu'
    results = model.predict(frame, conf=0.5, classes=vehicle_classes, device=0, verbose=False)

    annotated_frame = results[0].plot()
    cv2.imshow("Detecção de Veículos - GPU", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()