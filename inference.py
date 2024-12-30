import requests
import json
import os

# Configuration
IMAGE_PATH = "/home/sastra/Desktop/ObjectProperty/image.jpg"  # Path to the image
OUTPUT_FOLDER = "/home/sastra/Desktop/ObjectProperty/output"  # Path to save inference output
API_URL = "http://localhost:8000"
THRESHOLD = 0.35  # Threshold for inference

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
                    {"type": "text", "text": "mobile"},
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

def save_output(output_data, output_folder, image_id):
    """Save the inference output to a file."""
    output_path = os.path.join(output_folder, f"{image_id}_inference.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Inference output saved to {output_path}")

def main():
    try:
        # Step 1: Send the image
        print("Uploading image...")
        upload_response = send_image(IMAGE_PATH)
        image_id = upload_response["id"]
        print(f"Image uploaded successfully. ID: {image_id}")

        # Step 2: Send inference request
        print("Sending inference request...")
        inference_response = send_inference(image_id)
        print("Inference completed successfully.")

        # Step 3: Save inference output
        save_output(inference_response, OUTPUT_FOLDER, image_id)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
