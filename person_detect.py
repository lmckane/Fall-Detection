import cv2
import numpy as np
import os
import time
import argparse

# Force OpenCV to use GTK backend
os.environ["QT_QPA_PLATFORM"] = "xcb"

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

def draw_boxes(frame, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

    # Set the display window size
    cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Detection', 1280, 720)  # Adjust this size as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (416, 416))

        try:
            boxes = detect_persons(small_frame, net, output_layers)
            
            # Scale boxes to fit the original frame size
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            scaled_boxes = [[int(x * scale_x), int(y * scale_y), 
                             int(w * scale_x), int(h * scale_y)] for x, y, w, h in boxes]
            
            if not args.headless:
                display_frame = frame.copy()  # Create a copy for display
                display_frame = draw_boxes(display_frame, scaled_boxes)
                cv2.putText(display_frame, f"Persons: {len(boxes)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Person Detection', display_frame)
            
            print(f"Persons detected: {len(boxes)}")
            
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
    parser = argparse.ArgumentParser(description="Person Detection on Raspberry Pi")
    parser.add_argument("--stream_url", type=str, default="rtsp://admin:1adctester@172.30.3.184:554/s1", help="RTSP stream URL")
    parser.add_argument("--tiny", action="store_true", help="Use YOLOv3-tiny model")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    args = parser.parse_args()
    
    main(args)