import requests
import json
import os
import numpy as np
import PIL.Image
from utils.predictor import Predictor
import matplotlib.pyplot as plt

# Configuration
IMAGE_PATH = "/home/sastra/Desktop/ObjectProperty/image.jpg"  # Path to the image
OUTPUT_FOLDER = "/home/sastra/Desktop/ObjectProperty/output"  # Path to save inference output
API_URL = "http://localhost:8000"
THRESHOLD = 0.35  # Threshold for inference
SAM_IMAGE_ENCODER = "data/resnet18_image_encoder.engine"
SAM_MASK_DECODER = "data/mobile_sam_mask_decoder.engine"

# Initialize the Nano SAM predictor
predictor = Predictor(SAM_IMAGE_ENCODER, SAM_MASK_DECODER)

def send_image(image_path):
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

def send_inference(image_id):
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

def process_with_nano_sam(image_path, bounding_boxes):
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

def save_output(output_data, output_folder, image_id):
    """Save the inference output to a file."""
    output_path = os.path.join(output_folder, f"{image_id}_inference.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Inference output saved to {output_path}")

def visualize_results(image_path, masks):
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

    # Convert the overlay into an image and save
    result_image = PIL.Image.fromarray(overlay)
    result_image.save("segmentation_result_combined.png")

    print("Segmentation results saved as PNG.")

def main():
    print("Uploading image...")
    upload_response = send_image(IMAGE_PATH)
    image_id = upload_response["id"]
    print(f"Image uploaded successfully. ID: {image_id}")

    # Step 2: Send inference request
    print("Sending inference request...")
    inference_response = send_inference(image_id)
    print("Inference completed successfully.")

    # Extract bounding boxes from the response
    bounding_boxes_data = inference_response["choices"][0]["message"]["content"].get("boundingBoxes", [])

    # Retrieve only the 'bboxes' values
    bounding_boxes = [item["bboxes"] for item in bounding_boxes_data]
    print(bounding_boxes)
    # Step 3: Process with Nano SAM (segment detected objects)
    print("Processing with Nano SAM...")
    masks = process_with_nano_sam(IMAGE_PATH, bounding_boxes[0])

    # Step 4: Visualize and save the results
    visualize_results(IMAGE_PATH, masks)

    # Step 5: Save the inference output
    save_output(inference_response, OUTPUT_FOLDER, image_id)


if __name__ == "__main__":
    main()
