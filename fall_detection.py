import cv2
import numpy as np
import os
import time
import argparse
from collections import deque

# Force OpenCV to use GTK backend
os.environ["QT_QPA_PLATFORM"] = "xcb"

class PersonTracker:
    def __init__(self, max_disappear=10, max_distance=50):
        self.next_person_id = 0
        self.persons = {}
        self.disappeared = {}
        self.max_disappear = max_disappear
        self.max_distance = max_distance

    def register(self, centroid):
        self.persons[self.next_person_id] = centroid
        self.disappeared[self.next_person_id] = 0
        self.next_person_id += 1

    def deregister(self, person_id):
        del self.persons[person_id]
        del self.disappeared[person_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappear:
                    self.deregister(person_id)
            return self.persons

        centroids = np.array([((x + w // 2), (y + h // 2)) for (x, y, w, h) in boxes])

        if len(self.persons) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i])
        else:
            person_ids = list(self.persons.keys())
            previous_centroids = np.array(list(self.persons.values()))

            D = np.linalg.norm(previous_centroids[:, np.newaxis] - centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                person_id = person_ids[row]
                self.persons[person_id] = centroids[col]
                self.disappeared[person_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    person_id = person_ids[row]
                    self.disappeared[person_id] += 1
                    if self.disappeared[person_id] > self.max_disappear:
                        self.deregister(person_id)
            else:
                for col in unused_cols:
                    self.register(centroids[col])

        return self.persons

class FallDetector:
    def __init__(self, vertical_threshold=50, size_threshold=1.5, frame_threshold=5):
        self.vertical_threshold = vertical_threshold
        self.size_threshold = size_threshold
        self.frame_threshold = frame_threshold
        self.person_histories = {}

    def update(self, persons, boxes):
        fall_alerts = []

        for person_id, centroid in persons.items():
            if person_id not in self.person_histories:
                self.person_histories[person_id] = deque(maxlen=10)

            self.person_histories[person_id].append((centroid, boxes[person_id]))

            if len(self.person_histories[person_id]) >= self.frame_threshold:
                old_centroid, old_box = self.person_histories[person_id][-self.frame_threshold]
                new_centroid, new_box = self.person_histories[person_id][-1]

                vertical_change = abs(new_centroid[1] - old_centroid[1])
                size_change = (new_box[2] * new_box[3]) / (old_box[2] * old_box[3])

                if vertical_change > self.vertical_threshold and size_change > self.size_threshold:
                    fall_alerts.append((person_id, new_box))

        return fall_alerts

def load_yolo(config_path, weights_path, tiny=True):
    print(f"Loading YOLO{'v3-tiny' if tiny else 'v3'}...")
    net = cv2.dnn.readNet(weights_path, config_path)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    try:
        output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_persons(frame, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  # 0 is the class ID for person
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [boxes[i] for i in indexes]

def draw_boxes(frame, boxes, fall_alerts):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for _, box in fall_alerts:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "FALL DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

def main(args):
    print(f"OpenCV version: {cv2.__version__}")
    
    config_path = "yolov3-tiny.cfg" if args.tiny else "yolov3.cfg"
    weights_path = "yolov3-tiny.weights" if args.tiny else "yolov3.weights"
    
    for file in [config_path, weights_path, "coco.names"]:
        if os.path.exists(file):
            print(f"{file} found")
        else:
            print(f"{file} not found")
            return

    net, classes, output_layers = load_yolo(config_path, weights_path, args.tiny)

    print(f"Connecting to stream: {args.stream_url}")
    cap = cv2.VideoCapture(args.stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fall Detection', 1280, 720)

    person_tracker = PersonTracker()
    fall_detector = FallDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        small_frame = cv2.resize(frame, (416, 416))

        try:
            boxes = detect_persons(small_frame, net, output_layers)
            
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            scaled_boxes = [[int(x * scale_x), int(y * scale_y), 
                             int(w * scale_x), int(h * scale_y)] for x, y, w, h in boxes]
            
            persons = person_tracker.update(scaled_boxes)
            fall_alerts = fall_detector.update(persons, {i: box for i, box in enumerate(scaled_boxes)})

            if not args.headless:
                display_frame = frame.copy()
                display_frame = draw_boxes(display_frame, scaled_boxes, fall_alerts)
                cv2.putText(display_frame, f"Persons: {len(boxes)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Fall Detection', display_frame)
            
            print(f"Persons detected: {len(boxes)}, Falls detected: {len(fall_alerts)}")
            
            for _ in fall_alerts:
                print("FALL DETECTED! Alert triggered.")
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print("Resources released. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall Detection for Elderly Care")
    parser.add_argument("--stream_url", type=str, default="rtsp://admin:1adctester@172.30.3.184:554/s1", help="RTSP stream URL")
    parser.add_argument("--tiny", action="store_true", help="Use YOLOv3-tiny model")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    args = parser.parse_args()
    
    main(args)