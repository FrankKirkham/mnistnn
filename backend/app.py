from flask import Flask, request, jsonify
from flask_cors import CORS
from services import process, classify
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/classify', methods=['POST'])
def classify_upload():
    # Check that an image was given
    if "image" not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Convert the file to a useable PIL image
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))

    # Find the numbers in the image, then cut them out and turn them 
    # to one big tensor plus save the locations of the numbers
    tensor, locations = process.to_MNIST_tensor(image)

    # Classify these tensors
    results = classify.find_percentages(tensor)

    # Return a list of classified numbers, with their positions in the image
    final = []
    for result, location in zip(results, locations):
        dict = {
            'results': result,
            'location': location
        }
        final.append(dict)

    print(final)

    return final

if __name__ == '__main__':
    app.run(debug=True)