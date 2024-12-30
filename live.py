import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
from cv_bridge import CvBridge  # Convert ROS image messages to OpenCV images
from utils.predictor import Predictor
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000"
THRESHOLD = 0.35  # Threshold for inference
SAM_IMAGE_ENCODER = "data/resnet18_image_encoder.engine"
SAM_MASK_DECODER = "data/mobile_sam_mask_decoder.engine"

# Initialize the Nano SAM predictor
predictor = Predictor(SAM_IMAGE_ENCODER, SAM_MASK_DECODER)

# Initialize CvBridge for image conversion
bridge = CvBridge()

class RealsenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        
        # Subscribe to the RGB image topic (update topic names if needed)
        self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        # Optionally, subscribe to depth image topic as well
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

    def rgb_callback(self, msg):
        """Callback for RGB image."""
        try:
            # Convert ROS image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Process the image (e.g., save, display, send to server)
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
    
    def depth_callback(self, msg):
        """Callback for Depth image."""
        try:
            # Convert ROS depth image message to OpenCV image (16U)
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Process the depth image (optional)
            pass  # Add depth image processing if required
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def process_image(self, color_image):
        """Process the captured image (e.g., send to server, run inference)."""
        # Save the frame temporarily for processing
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, color_image)

        try:
            # Step 1: Upload the image
            print("Uploading image...")
            upload_response = self.send_image(temp_image_path)
            image_id = upload_response["id"]
            print(f"Image uploaded successfully. ID: {image_id}")

            # Step 2: Send inference request
            print("Sending inference request...")
            inference_response = self.send_inference(image_id)
            print("Inference completed successfully.")

            # Extract bounding boxes from the response
            bounding_boxes_data = inference_response["choices"][0]["message"]["content"].get("boundingBoxes", [])
            bounding_boxes = [item["bboxes"] for item in bounding_boxes_data]

            # Step 3: Process with Nano SAM
            print("Processing with Nano SAM...")
            masks = self.process_with_nano_sam(temp_image_path, bounding_boxes[0])

            # Step 4: Visualize the results
            self.visualize_results(temp_image_path, masks)

        except Exception as e:
            print(f"Error during processing: {e}")
    
    def send_image(self, image_path):
        """Send the image to the `/files` endpoint."""
        url = f"{API_URL}/files"
        files = {
            "file": open(image_path, "rb"),
        }
        data = {
            "purpose": "vision",
            "media_type": "image",
        }
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            return response.json()  # Return JSON response
        else:
            raise Exception(f"Failed to upload image. Status code: {response.status_code}, Response: {response.text}")
    
    def send_inference(self, image_id):
        """Send inference request to the `/inference` endpoint."""
        url = f"{API_URL}/inference"
        input_json = {
            "model": "Grounding-Dino",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "phone in hand"},
                        {"type": "media_url", "media_url": {"url": f"data:image/jpeg;asset_id,{image_id}"}}
                    ]
                }
            ],
            "threshold": THRESHOLD,
        }
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(input_json))
        if response.status_code == 200:
            return response.json()  # Return JSON response
        else:
            raise Exception(f"Failed to perform inference. Status code: {response.status_code}, Response: {response.text}")
    
    def process_with_nano_sam(self, image_path, bounding_boxes):
        """Process the image with Nano SAM using the bounding boxes from Grounding DINO."""
        image = PIL.Image.open(image_path)
        predictor.set_image(image)

        masks = []
        for bbox in bounding_boxes:
            # Convert bounding box to points
            points = np.array([
                [bbox[0], bbox[1]],  # top-left corner
                [bbox[2]+bbox[0], bbox[3]+bbox[1]]   # bottom-right corner
            ])
            # Predict the mask for each bounding box
            mask, _, _ = predictor.predict(points, np.array([2, 3]))  # Assuming point_labels are [2, 3]
            masks.append(mask)
        return masks
    
    def visualize_results(self, image_path, masks):
        """Apply all masks to the image and save the result as a single PNG."""
        image = PIL.Image.open(image_path)
        image_np = np.array(image)

        # Create a mask overlay where all masks will be applied
        overlay = np.zeros_like(image_np)

        for i, mask in enumerate(masks):
            mask = (mask[0, 0] > 0).detach().cpu().numpy()

            # Find the bounding box of the mask (non-zero area)
            y_indices, x_indices = np.where(mask > 0)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # Apply the mask to the overlay image (you can use any color here)
            overlay[y_min:y_max, x_min:x_max][mask[y_min:y_max, x_min:x_max]] = image_np[y_min:y_max, x_min:x_max][mask[y_min:y_max, x_min:x_max]]

        # Convert the overlay into an image and show
        result_image = PIL.Image.fromarray(overlay)
        result_image.show()

def main():
    rclpy.init()
    realsense_subscriber = RealsenseSubscriber()
    rclpy.spin(realsense_subscriber)

    # Destroy the node explicitly
    realsense_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
