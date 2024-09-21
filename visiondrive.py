import sys
import cv2
import torch
import numpy as np
import easyocr
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


class VehicleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load pre-trained YOLOv5 model with GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=self.device)

        # Load EasyOCR reader
        self.ocr_reader = easyocr.Reader(['en'])

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Video capture object
        self.capture = None
        self.video_stream = ""

        # Vehicle classes in COCO dataset (cars, trucks, buses)
        self.vehicle_classes = [2, 5, 7]  # Class IDs for cars, buses, trucks in COCO dataset

        # Tracking vehicle speed
        self.prev_positions = {}  # To track previous positions of vehicles
        self.frame_rate = 60  # Default frame rate
        self.pixels_to_meter_ratio = 0.03 # Conversion ratio, adjust based on scene

        # Limit OCR execution frequency
        self.ocr_frequency = 5
        self.frame_count = 0

    def initUI(self):
        # Window properties
        self.setWindowTitle("VisionDrive - ANPR System")
        self.setGeometry(100, 100, 1280, 720)

        # Start button
        self.start_button = QPushButton("Start The Video", self)
        self.start_button.setGeometry(10, 10, 150, 40)
        self.start_button.clicked.connect(self.start_video)

        # Stop button
        self.stop_button = QPushButton("Stop The Video", self)
        self.stop_button.setGeometry(10, 60, 150, 40)
        self.stop_button.clicked.connect(self.stop_video)

        # Load video button
        self.load_button = QPushButton("Load The Video", self)
        self.load_button.setGeometry(10, 110, 150, 40)
        self.load_button.clicked.connect(self.load_video)

        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setGeometry(200, 10, 1000, 700)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

    def start_video(self):
        if not self.video_stream:
            # Open webcam if no video stream is loaded
            self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.VideoCapture(self.video_stream)

        # Get video frame rate for speed calculation
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        if self.frame_rate == 0:
            self.frame_rate = 60  # Default to 30 FPS if video file frame rate cannot be retrieved
        print(f"Frame rate of the video: {self.frame_rate} FPS")

        self.timer.start(30)

    def stop_video(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()

    def load_video(self):
        # Open file dialog to select video
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_stream = file_name

    def update_frame(self):
        # Capture frame-by-frame
        ret, frame = self.capture.read()

        if ret:
            # Resize frame to improve performance
            frame_resized = cv2.resize(frame, (640, 480))

            # Process the frame and detect vehicles
            detected_frame, vehicle_speeds = self.detect_vehicles_and_calculate_speed(frame_resized)

            # Convert the frame to QImage for PyQt display
            rgb_image = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Resize the QImage to fit QLabel dimensions (full screen)
            scaled_image = qt_image.scaled(self.video_label.width(), self.video_label.height(),
                                           aspectRatioMode=Qt.KeepAspectRatio)

            # Display the scaled image in the label
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
        else:
            self.stop_video()

    def detect_vehicles_and_calculate_speed(self, frame):
        # Run YOLO model to detect objects
        results = self.model(frame)

        current_positions = {}
        vehicle_speeds = {}

        # Filter for vehicle-related classes (car, truck, bus)
        for i, (*box, conf, cls) in enumerate(results.xyxy[0]):
            if int(cls) in self.vehicle_classes:
                x1, y1, x2, y2 = map(int, box)
                label = f'{self.model.names[int(cls)]} {conf:.2f}'
                vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate vehicle center

                # Generate unique vehicle ID based on detected object index
                vehicle_id = f'vehicle_{i}'

                # Speed calculation logic
                if vehicle_id in self.prev_positions:
                    prev_center = self.prev_positions[vehicle_id]
                    distance = np.linalg.norm(np.array(vehicle_center) - np.array(prev_center))
                    print(f"Vehicle {vehicle_id} moved {distance:.2f} pixels.")

                    # Ensure frame_rate is correct before calculation
                    if self.frame_rate > 0:
                        speed = (distance * self.pixels_to_meter_ratio) / (1 / self.frame_rate)  # meters per second
                        vehicle_speeds[vehicle_id] = speed * 3.6  # Convert m/s to km/h
                        print(f"Vehicle {vehicle_id} speed: {vehicle_speeds[vehicle_id]:.2f} km/h")
                    else:
                        vehicle_speeds[vehicle_id] = 0.0  # If frame rate is incorrect, set speed to 0
                else:
                    # If no previous position, assume 0 speed
                    vehicle_speeds[vehicle_id] = 0.0

                # Update current position
                current_positions[vehicle_id] = vehicle_center

                # Draw bounding box and speed on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                speed_text = f'{label} Speed: {vehicle_speeds.get(vehicle_id, 0):.2f} km/h'
                cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Detect license plate using EasyOCR every N frames
                if self.frame_count % self.ocr_frequency == 0:
                    license_plate_region = frame[y1:y2, x1:x2]  # Crop the vehicle region
                    ocr_results = self.ocr_reader.readtext(license_plate_region)

                    # If text detected, print the number plate on the frame
                    for (bbox, text, prob) in ocr_results:
                        if prob > 0.5:  # Filter based on confidence
                            print(f"Detected license plate: {text}")
                            cv2.putText(frame, f"Plate: {text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Identify vehicle color
                vehicle_color = self.identify_vehicle_color(frame[y1:y2, x1:x2])
                cv2.putText(frame, f"Color: {vehicle_color}", (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Update the previous positions for the next frame
        self.prev_positions = current_positions

        # Increment the frame counter
        self.frame_count += 1

        return frame, vehicle_speeds

    def identify_vehicle_color(self, vehicle_region):
        """Identify the color of the vehicle based on the most dominant color in the region."""
        # Convert the vehicle region to RGB format
        vehicle_region_rgb = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2RGB)

        # Reshape the region to a list of pixels
        pixels = vehicle_region_rgb.reshape((-1, 3))

        # Use k-means clustering to find the dominant color
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]

        # Convert the dominant color to integer values
        dominant_color = [int(c) for c in dominant_color]

        # Map the dominant color to a human-readable color
        color_name = self.get_color_name(dominant_color)
        return color_name

    def get_color_name(self, rgb_color):
        """Map the RGB value to a human-readable color name."""
        colors = {
            (255, 0, 0): 'Red',
            (0, 255, 0): 'Green',
            (0, 0, 255): 'Blue',
            (255, 255, 0): 'Yellow',
            (0, 255, 255): 'Cyan',
            (255, 0, 255): 'Magenta',
            (255, 255, 255): 'White',
            (0, 0, 0): 'Black',
            (128, 128, 128): 'Gray',
            (192, 192, 192): 'Silver',
            (128, 0, 0): 'Maroon',
            (0, 128, 0): 'Dark Green',
            (0, 0, 128): 'Navy',
            (128, 128, 0): 'Olive',
            (255, 165, 0): 'Orange',
            (75, 0, 130): 'Indigo',
            (238, 130, 238): 'Violet'
        }

        # Find the closest color by Euclidean distance
        closest_color = min(colors.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(rgb_color)))
        return colors[closest_color]

    # Dynamically resize QLabel when window is resized
    def resizeEvent(self, event):
        self.video_label.resize(self.width() - 200, self.height() - 100)  # Adjust size according to window
        super().resizeEvent(event)


def main():
    app = QApplication(sys.argv)
    main_window = VehicleDetectionApp()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

