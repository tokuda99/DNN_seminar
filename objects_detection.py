# ライブラリのインポート
import cv2
from ultralytics import YOLO
from argparse import ArgumentParser

# 引数の取得
parser = ArgumentParser()
parser.add_argument("--classes", type=int, nargs="+", default=[i for i in range(80)])
parser.add_argument("--conf", type=float, default=0.25)
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# YOLOv11のモデルをロード
model = YOLO("yolo11n.pt") 

# Webカメラの起動
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOで物体検出を行う
    results = model(frame, classes=parser.parse_args().classes, conf=parser.parse_args().conf)

    # 結果をフレームに描画して表示
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    # ウィンドウに描画した結果を表示
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()