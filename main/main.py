# Import the necessary libraries
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker_resultados import Tracker

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the area of interest with external and internal lines (long sides of the rectangle)
left_line_x = 325  # Línea roja (izq)
right_line_x = 331 # Línea amarilla (dcha), pero no la necesitamos para detectar entrada

# Define the area of interest (blue rectangle) in the correct order of coordinates
# The area is defined by four points (rectangular region) as (x, y) pairs
area1 = np.array([(325, 723), (331, 723), (561, 121), (555, 121)], np.int32)  # Rectángulo completo (azul)

# Open the train station video
cap = cv2.VideoCapture('1_TRI_C1.mkv')

# Get FPS from the video
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)

# Read the COCO class file
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0  # Frame counter
tracker = Tracker()

people_inside_area = set()  # IDs dentro del área entre líneas
crossed_red_line = set()  # Personas que han cruzado la línea roja
entering = set()  # Personas que han entrado al tranvía
exiting = set()  # Personas que han salido del tranvía
temp_list = {}  # Lista temporal de personas que han entrado en el área azul

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('metropolitano_output.mp4', fourcc, 5.0, (1280, 720))  # Output at 5 FPS

frame_skip = 5  # Skip every 5 frames

# Main loop to process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # Skip frames to reduce processing load
    if count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (1280, 720))
    results = model.predict(frame, conf=0.5)
    px = pd.DataFrame(results[0].boxes.data).astype("float")
    list1 = []

    # Draw the area of interest on the frame
    cv2.polylines(frame, [area1], True, (255, 0, 0), 2)  # Rectángulo completo (azul)
    cv2.line(frame, (area1[0][0], area1[1][1]), (area1[3][0], area1[2][1]), (0, 0, 255), 2) 
    #cv2.line(frame, (area1[1][0], area1[0][1]), (area1[2][0], area1[2][1]), (0, 0, 255), 2)  # Línea roja

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        c = class_list[int(row[5])]
        if 'person' in c:
            list1.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list1)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        feet_x = (x3 + x4) // 2  # Coordenada central inferior
        feet_y = y4

        # Determinar si la persona está dentro del área azul
        inside_area = cv2.pointPolygonTest(np.array(area1, np.int32), (feet_x, feet_y), False) >= 0

        # Cambiar el color del recuadro según la posición
        if inside_area and id not in people_inside_area:
            people_inside_area.add(id)  # Añadir el id cuando entra en el área
            print(f"Persona con ID {id} está en el área de interés")

        # Coordenadas de la línea roja (comienzo y fin)
        line_start = (325, 723)  # Coordenada inicial de la línea roja
        line_end = (555, 121)  # Coordenada final de la línea roja

        # Calcular la pendiente de la línea roja
        m = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])

        # Verificar si la persona cruza la línea roja
        line_y_at_x = m * (feet_x - line_start[0]) + line_start[1]

        # Si la persona cruza la línea roja, la añadimos a crossed_red_line
        if abs(feet_y - line_y_at_x) < 10 and id not in crossed_red_line:
            crossed_red_line.add(id)
            print(f"Persona con ID {id} ha cruzado la línea roja")

        # Si la persona cumple la condición de haber entrado en el área y cruzado la línea roja
        if id in people_inside_area and id in crossed_red_line and id not in entering:
            entering.add(id)
            print(f"Persona con ID {id} entrando")

        # Si la persona cruza la línea roja después de entrar en el área azul, registramos como entrada
        if feet_x > left_line_x and id in temp_list and temp_list[id] == 'entered_area' and id not in entering:
            entering.add(id)
            exiting.add(id)  # Marcar como saliendo también
            del temp_list[id]  # Eliminar de la lista temporal una vez ha cruzado la línea roja

        # Añadir la persona a la lista temporal si entra en el área azul
        if feet_x < right_line_x and id not in temp_list and inside_area:
            temp_list[id] = 'entered_area'  # Guardamos que ha entrado en el área azul

        # Dibujar detecciones con el color definido
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Mostrar ID
        cv2.circle(frame, (feet_x, feet_y), 5, (255, 0, 255), -1)

    # Display counts
    cv2.putText(frame, f'Entering: {len(entering)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f'Exiting: {len(exiting)}', (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(wait_time) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

