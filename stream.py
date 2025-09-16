import cv2
import argparse
import numpy as np
import math
from ultralytics import YOLO
import supervision as sv
import threading
from flask import Flask, Response, render_template, request, jsonify
import time
import os
import datetime
import uuid
import requests
import json
import wikipedia

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live web stream")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int,
        help="Resolution of the webcam feed"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="webcam",
        help="Path to the input file (image, video, or 'webcam')"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port for the web server"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host address for the web server"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="captured_images",
        help="Directory to save captured images"
    )
    args = parser.parse_args()
    return args

def calculate_angle(frame, midpoint):
    origin_x, origin_y = frame.shape[1] // 2, frame.shape[0] - 1
    angle_radians = math.atan2(midpoint[0] - origin_x, origin_y - midpoint[1])
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def draw_annotations(frame, midpoint, angle):
    # Draw line from horizontal center of the frame to midpoint
    frame_center_x = frame.shape[1] // 2
    cv2.line(frame, (frame_center_x, frame.shape[0]), (int(midpoint[0]), int(midpoint[1])), (0, 255, 0), 2)
    
    # Draw circle at the midpoint of the bounding box
    cv2.circle(frame, (int(midpoint[0]), int(midpoint[1])), 4, (0, 255, 0), -1)
    
    # Display the angle on the frame
    cv2.putText(frame, "{:.2f} degrees".format(angle), (int(midpoint[0]), int(midpoint[1]) + 20), 
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

# Function to get information about an object using Wikipedia
def get_object_info(object_name):
    try:
        # Search Wikipedia for the object
        search_results = wikipedia.search(object_name, results=1)
        
        if not search_results:
            return f"No Wikipedia information found for {object_name}."
        
        # Get the page summary
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=2)
        
        return summary
    
    except wikipedia.exceptions.DisambiguationError as e:
        # If disambiguation is needed, use the first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
            return summary
        except:
            return f"Multiple possible Wikipedia entries for {object_name}, but couldn't retrieve information."
    
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for {object_name}."
    
    except Exception as e:
        return f"Error fetching information: {str(e)}"

# Global variables for frame sharing between threads
output_frame = None
raw_frame = None  # Store the raw frame before annotations
lock = threading.Lock()
model = None
box_annotator = None
save_dir = "captured_images"
last_captured_image = None
inference_image = None
inference_result = None
inference_lock = threading.Lock()
object_info = {}  # Dictionary to store object information

def process_frames(args):
    global output_frame, raw_frame, lock, model, box_annotator, save_dir
    
    frame_width, frame_height = args.webcam_resolution
    save_dir = args.save_dir

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.input == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load the YOLO model 
    model_path = os.path.expanduser("/home/sid/trial/trained models/2-model disease .pt")
    model = YOLO(model_path)

    # Initialize annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.input == "webcam":
                    print("Error reading from webcam.")
                    time.sleep(1)  # Wait before retrying
                    continue
                elif args.input.endswith(('.jpg', '.jpeg', '.png')):
                    # For images, keep running the inference until the program is killed
                    cap = cv2.VideoCapture(args.input)
                    continue
                else:
                    # For videos, loop back to the beginning
                    cap = cv2.VideoCapture(args.input)
                    continue

            # Store the raw frame for capture functionality
            with lock:
                raw_frame = frame.copy()
                output_frame = frame.copy()  # Now just using the raw frame for output

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        print("Video capture released.")

def generate_frames():
    global output_frame, lock
    while True:
        # Wait until a frame is available
        if output_frame is None:
            time.sleep(0.1)
            continue
            
        # Encode frame as JPEG
        with lock:
            if output_frame is not None:
                encoded_frame = cv2.imencode('.jpg', output_frame)[1].tobytes()
            
        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

def generate_inference_frames():
    global inference_result, inference_lock
    while True:
        # Wait until an inference result is available
        with inference_lock:
            current_result = inference_result
            
        if current_result is None:
            # If no inference result is available, yield a placeholder image
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, 
                "No inference result available", 
                (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            encoded_frame = cv2.imencode('.jpg', placeholder)[1].tobytes()
        else:
            # Encode the inference result as JPEG
            encoded_frame = cv2.imencode('.jpg', current_result)[1].tobytes()
            
        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')
        
        # Sleep to control frame rate
        time.sleep(0.1)

def capture_and_save_image():
    global raw_frame, lock, save_dir, last_captured_image
    
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # Save the current raw frame
    with lock:
        if raw_frame is not None:
            cv2.imwrite(filepath, raw_frame)
            last_captured_image = filepath
            print(f"Image saved to {filepath}")
            return filepath
    
    return None

def run_inference_on_image(image_path):
    global model, box_annotator, inference_image, inference_result, inference_lock, object_info
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Store the original image
    inference_image = image.copy()
    
    # Perform object detection
    result = model(image, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    
    # Reset object_info dictionary
    object_info = {}
    
    # Get unique detected classes
    detected_classes = set()
    for i in range(len(detections.xyxy)):
        class_id = detections.class_id[i]
        class_name = model.model.names[class_id]
        detected_classes.add(class_name)
    
    # Fetch information for each unique class
    for class_name in detected_classes:
        # Only fetch information if we haven't already done so
        if class_name not in object_info:
            info = get_object_info(class_name)
            object_info[class_name] = info
            print(f"Fetched information for {class_name}: {info}")
    
    # Prepare labels for the bounding boxes
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    
    # Annotate the frame with bounding boxes and labels
    annotated_image = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections,
        labels=labels
    )
    
    # Process detections for angle calculation
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Calculate midpoint of the bounding box
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        midpoint = (midpoint_x, midpoint_y)
        
        # Calculate the angle from the horizontal center
        # angle = calculate_angle(annotated_image, midpoint)
        
        # Draw annotations
        # draw_annotations(annotated_image, midpoint, angle)
    
    # Update the inference result
    with inference_lock:
        inference_result = annotated_image.copy()
    
    return True

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inference_feed')
def inference_feed():
    return Response(generate_inference_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global object_info
    image_path = capture_and_save_image()
    if image_path:
        # Run inference on the captured image
        success = run_inference_on_image(image_path)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Image captured and saved to {image_path}',
                'image_path': image_path,
                'object_info': object_info
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to run inference on the captured image'
            })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to capture image'
        })

@app.route('/object_info')
def get_object_info_route():
    global object_info
    return jsonify(object_info)

def create_templates_directory():
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create the HTML template file
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Object Detection with Wikipedia</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            color: #333;
        }
        .container {
            max-width: 1280px;
            margin: 0 auto;
        }
        .video-container {
            background-color: #ddd;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            display: inline-block;
            margin-right: 10px;
            vertical-align: top;
            width: 45%;
        }
        .video-label {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .button-container {
            margin: 20px 0;
        }
        .capture-btn {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .capture-btn:hover {
            background-color: #45a049;
        }
        .capture-btn:active {
            background-color: #3e8e41;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
                .status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .info-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: left;
        }
        .info-title {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
            color: #333;
        }
        .object-info {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .object-info:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .object-name {
            font-weight: bold;
            margin-bottom: 5px;
            color: #2c3e50;
        }
        .object-description {
            color: #555;
            line-height: 1.5;
        }
        .wiki-link {
            color: #3498db;
            text-decoration: none;
            display: block;
            margin-top: 5px;
            font-size: 0.9em;
        }
        .wiki-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLO Object Detection with Wikipedia Integration</h1>
        
        <div class="button-container">
            <button id="captureBtn" class="capture-btn">Capture Image & Run Inference</button>
        </div>
        
        <div id="status" class="status"></div>
        
        <div class="flex-container">
            <div class="video-container">
                <div class="video-label">Live Stream (Raw Feed)</div>
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
            
            <div class="video-container">
                <div class="video-label">Inference Result (With Annotations)</div>
                <img src="{{ url_for('inference_feed') }}" alt="Inference Result">
            </div>
        </div>
        
        <div id="objectInfoContainer" class="info-container" style="display: none;">
            <div class="info-title">Object Information from Wikipedia:</div>
            <div id="objectInfo"></div>
        </div>
    </div>

    <script>
        document.getElementById('captureBtn').addEventListener('click', function() {
            // Disable button during processing
            const button = this;
            button.disabled = true;
            button.textContent = 'Processing...';
            
            // Display status message
            const statusElement = document.getElementById('status');
            statusElement.className = 'status';
            statusElement.textContent = 'Capturing image and running inference...';
            statusElement.style.display = 'block';
            
            // Send request to server
            fetch('/capture', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                // Reset button
                button.disabled = false;
                button.textContent = 'Capture Image & Run Inference';
                
                // Display status message
                if (data.status === 'success') {
                    statusElement.className = 'status success';
                    statusElement.textContent = data.message;
                    
                    // Display object information
                    displayObjectInfo(data.object_info);
                } else {
                    statusElement.className = 'status error';
                    statusElement.textContent = data.message;
                }
            })
            .catch(error => {
                // Reset button
                button.disabled = false;
                button.textContent = 'Capture Image & Run Inference';
                
                // Display error message
                statusElement.className = 'status error';
                statusElement.textContent = 'Error: ' + error.message;
            });
        });
        
        function displayObjectInfo(objectInfo) {
            const container = document.getElementById('objectInfoContainer');
            const infoDiv = document.getElementById('objectInfo');
            
            // Clear previous content
            infoDiv.innerHTML = '';
            
            // Check if we have any object information
            if (Object.keys(objectInfo).length === 0) {
                infoDiv.innerHTML = '<p>No objects detected or no information available.</p>';
                container.style.display = 'block';
                return;
            }
            
            // Create HTML for each object's information
            for (const [objectName, description] of Object.entries(objectInfo)) {
                const objectDiv = document.createElement('div');
                objectDiv.className = 'object-info';
                
                const nameElem = document.createElement('div');
                nameElem.className = 'object-name';
                nameElem.textContent = objectName;
                
                const descElem = document.createElement('div');
                descElem.className = 'object-description';
                descElem.textContent = description;
                
                // Add Wikipedia link
                const linkElem = document.createElement('a');
                linkElem.className = 'wiki-link';
                linkElem.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(objectName)}`;
                linkElem.target = '_blank';
                linkElem.textContent = 'Read more on Wikipedia';
                
                objectDiv.appendChild(nameElem);
                objectDiv.appendChild(descElem);
                objectDiv.appendChild(linkElem);
                infoDiv.appendChild(objectDiv);
            }
            
            // Show the container
            container.style.display = 'block';
        }
    </script>
</body>
</html>
        ''')

def main():
    args = parse_arguments()
    
    # Create the templates directory and HTML file
    create_templates_directory()
    
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_frames, args=(args,))
    video_thread.daemon = True
    video_thread.start()
    
    # Start the Flask web server
    print(f"Starting web server at http://{args.host}:{args.port}/")
    print(f"Captured images will be saved to: {args.save_dir}")
    print("Using Wikipedia to retrieve object information")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()