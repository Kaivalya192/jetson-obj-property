import ollama
# prompt=f"""
#         Identify the object in the image and provide the following material properties. Respond only with the required values, no comments or notes:

#         - "Object name": (one or two words)
#         - "Material": (one or two words)
#         - "Coefficient of friction": (numeric value)
#         - "sDimensions (L x W x H in cm)": (3 numeric values separated by commas)
#         - "Weight (in grams)": (numeric value)

#         """
prompt=f"""
        Identify the object in the image and provide the following material properties.ignore black background.Respond only with the required values, no comments or notes:

        - "Object name": (one or two words)
        - "Material": (one or two words)
        - "Coefficient of friction": (numeric value)
        - "sDimensions (L x W x H in cm)": (3 numeric values separated by commas)
        - "Weight (in grams)": (numeric value)

        """

file_path = "segmentation_result_combined.png"

stream = ollama.generate(
    model="llava:7b",
    prompt=prompt,
    images=[file_path],
    stream=True
)

for chunk in stream:
    print(chunk['response'], end='')
