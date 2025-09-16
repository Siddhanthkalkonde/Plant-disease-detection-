import cv2
import argparse
import numpy as np
import math
from ultralytics import YOLO
import supervision as sv
# from arduinoSerialCom import ArduinoSerial

# Initialize ArduinoSerial for communication
# arduino = ArduinoSerial(115200, '/dev/ttyACM0')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
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

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    if args.input == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load the YOLO model 
    model = YOLO("/home/sid/trial/trained models/palmdetection.pt")

    # Initialize annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    try:
        # Open the serial port before entering the loop
        # arduino.open_port()

        while True:
            ret, frame = cap.read()
            if not ret:
                if args.input == "webcam":
                    print("Error reading from webcam.")
                    break
                elif args.input.endswith(('.jpg', '.jpeg', '.png')):
                    # For images, keep running the inference until the program is killed
                    cap = cv2.VideoCapture(args.input)
                    continue
                else:
                    # For videos, break the loop when the video ends
                    break

            # Perform object detection
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)

            # Prepare labels for the bounding boxes
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            # Annotate the frame with bounding boxes and labels
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections
            )

            # Process detections for angle calculation and serial communication
            angles = []  # Buffer for angles to send
            for i in range(len(detections.xyxy)):
                detected_class_name = model.model.names[detections.class_id[i]]
                box = detections.xyxy[i]
                confidence = detections.confidence[i]

                # Extract bounding box coordinates
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Ensure integer coordinates

                # Calculate midpoint of the bounding box
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                midpoint = (midpoint_x, midpoint_y)

                # Calculate the angle from the horizontal center
                angle = calculate_angle(frame, midpoint)
                angles.append(f"{angle:.2f}")

                # Draw annotations
                draw_annotations(frame, midpoint, angle)

            # Send angles to Arduino; default to "0" if no detections
            # if angles:
            #     angles_string = ' '.join(angles)
            # else:
            #     angles_string = "-60"
            
            # try:
            #     arduino.send_data(angles_string)
            # except Exception as e:
            #     arduino.send_data("-60")
            #     print(f"An error occurred while sending data: {e}")

            # Display the annotated frame
            cv2.imshow("yolov8", frame)

            # Exit loop on 'ESC' key press
            if cv2.waitKey(30) == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Ensure the final string "-60" is sent before exiting
        # arduino.send_data("-60")
        cap.release()
        cv2.destroyAllWindows()
        # arduino.close_port()  # Close the serial port properly

if __name__ == "__main__":
    main()