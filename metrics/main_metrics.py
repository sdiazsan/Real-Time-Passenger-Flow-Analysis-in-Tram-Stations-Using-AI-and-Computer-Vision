import time
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from tracker_resultados import Tracker
from sklearn.metrics import precision_score, recall_score, f1_score

# Crear carpeta de resultados
output_folder = "resultados"
os.makedirs(output_folder, exist_ok=True)

# Modelos YOLO a comparar
models = {
    "YOLOv8n": YOLO('yolov8n.pt'),
    "YOLOv8s": YOLO('yolov8s.pt'),
    "YOLOv8m": YOLO('yolov8m.pt'),
    "YOLOv8l": YOLO('yolov8l.pt'),
    "YOLOv5n": YOLO('yolov5n.pt'),
    "YOLOv5s": YOLO('yolov5s.pt'),
    "YOLOv5m": YOLO('yolov5m.pt'),
    "YOLOv5l": YOLO('yolov5l.pt'),
    "YOLOv3": YOLO('yolov3.pt')
}

# Variables para almacenar resultados de cada modelo
metrics_results = {}

# Leer video
video_path = '1_TRI_C1.mkv'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Procesar cada modelo
for model_name, model in models.items():
    print(f"\nProcesando con {model_name}...")

    tracker = Tracker()
    detections_per_frame = []
    tracking_accuracy = []
    frame_times = []
    entrances = []
    pred_classes = []
    gt_classes = []

    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video

    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        start_time = time.time()

        results = model.predict(frame, conf=0.5)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
        detections_per_frame.append(len(detections))

        bbox_list = [[int(x) for x in det[:4]] for det in detections]
        tracked_objects = tracker.update(bbox_list)
        tracking_accuracy.append(len(tracked_objects) / max(1, len(detections)))

        detected_entrances = len(tracker.center_points)
        entrances.append(detected_entrances)
        pred_classes.append(1 if detected_entrances > 0 else 0)

        end_time = time.time()
        frame_times.append(end_time - start_time)

    manually_labeled_entrances = []
    previous_count = 0
    for current_count in entrances:
        new_entries = max(0, current_count - previous_count)
        manually_labeled_entrances.append(new_entries)
        previous_count = current_count

    gt_classes = [1 if e > 0 else 0 for e in manually_labeled_entrances]

    avg_inference_time = np.mean(frame_times)
    avg_detections = np.mean(detections_per_frame)
    tracking_precision = np.mean(tracking_accuracy)
    precision = precision_score(gt_classes, pred_classes, average='macro', zero_division=0)
    recall = recall_score(gt_classes, pred_classes, average='macro', zero_division=0)
    f1 = f1_score(gt_classes, pred_classes, average='macro', zero_division=0)

    metrics_results[model_name] = {
        "Tiempo Inferencia": avg_inference_time,
        "Detecciones por Frame": avg_detections,
        "Precisión del Tracker": tracking_precision,
        "Precisión": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    plt.figure(figsize=(10, 6))
    plt.plot(detections_per_frame, label='Detecciones por Frame')
    plt.plot(entrances, label='Entradas Detectadas', color='green')
    plt.legend()
    plt.title(f'Métricas de Conteo - {model_name}')
    plt.xlabel('Frames')
    plt.ylabel('Cantidad')
    plt.savefig(f"{output_folder}/{model_name}_metricas.png")
    plt.close()

results_file = os.path.join(output_folder, "resultados_comparativos.txt")
with open(results_file, "w") as f:
    f.write("Comparación de Modelos YOLO (v3, v5, v8)\n")
    f.write("=" * 50 + "\n")
    for model_name, metrics in metrics_results.items():
        f.write(f"\nModelo: {model_name}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("=" * 50 + "\n")

# GRÁFICAS COMPARATIVAS
model_names = list(metrics_results.keys())
precision_values = [metrics_results[m]["Precisión"] for m in model_names]
recall_values = [metrics_results[m]["Recall"] for m in model_names]
f1_values = [metrics_results[m]["F1 Score"] for m in model_names]
inference_times = [metrics_results[m]["Tiempo Inferencia"] for m in model_names]
detections_per_frame = [metrics_results[m]["Detecciones por Frame"] for m in model_names]

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].bar(model_names, precision_values, color='blue', alpha=0.7, label='Precisión')
axs[0, 0].bar(model_names, recall_values, color='green', alpha=0.7, label='Recall', bottom=precision_values)
axs[0, 0].bar(model_names, f1_values, color='red', alpha=0.7, label='F1-Score', bottom=[p + r for p, r in zip(precision_values, recall_values)])
axs[0, 0].set_title("Precisión, Recall y F1-Score")
axs[0, 0].legend()

axs[0, 1].bar(model_names, inference_times, color='purple', alpha=0.7)
axs[0, 1].set_title("Tiempo Promedio de Inferencia (s)")

axs[1, 0].bar(model_names, detections_per_frame, color='orange', alpha=0.7)
axs[1, 0].set_title("Detecciones por Frame")

plt.tight_layout()
plt.savefig(f"{output_folder}/comparacion_modelos.png")
plt.show()

print(f"\nResultados guardados en {output_folder}")

