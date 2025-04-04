# # from flask import Flask

# # app = Flask(__name__)

# # @app.route('/')
# # def home():
# #     return 'Hello, World!'

# # @app.route('/about')
# # def about():
# #     return 'About'

# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import os
# import tensorflow as tf
# from io import BytesIO

# app = Flask(__name__)

# # Setup dummy generator to extract class labels
# dummy_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
# dummy_generator = dummy_datagen.flow_from_directory(
#     'dataset/train',  # same folder used for training
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode='categorical'
# )

# class_labels = list(dummy_generator.class_indices.keys())
# print("Detected class labels:", class_labels)

# # Load trained model
# model = load_model("lung5.keras")
# print("Model loaded successfully!")

# # Image preprocessing function
# def load_and_preprocess_image(img_file, target_size=(224, 224)):
#     img = image.load_img(img_file, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         file = BytesIO(file.read())  # convert FileStorage to BytesIO
        
#         img = load_and_preprocess_image(file, target_size=(224, 224))
#         predictions = model.predict(img)
#         predicted_class = int(np.argmax(predictions[0]))
#         predicted_label = class_labels[predicted_class]

#         return jsonify({
#             'predicted_class_index': predicted_class,
#             'predicted_label': predicted_label,
#             'confidence': float(np.max(predictions[0]))
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.route('/')
# def home():
#     return 'Hello, World!'


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Use the confirmed class labels directly
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
logger.info(f"Using class labels: {class_labels}")

# Load trained model
try:
    model = load_model("lung5.keras")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Image preprocessing function
def load_and_preprocess_image(img_file, target_size=(224, 224)):
    """
    Load and preprocess an image for model prediction
    
    Args:
        img_file: Image file object
        target_size: Target size for the image
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    img = image.load_img(img_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions on lung CT images"""
    # Check if image is included in request
    if 'image' not in request.files:
        logger.warning("Prediction request missing image file")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("Empty filename in prediction request")
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Convert FileStorage to BytesIO
        file = BytesIO(file.read())
        
        # Preprocess image
        img = load_and_preprocess_image(file, target_size=(224, 224))
        
        # Get predictions
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        predicted_label = class_labels[predicted_class]
        confidence = float(np.max(predictions[0]))
        
        logger.info(f"Prediction: {predicted_label} (Confidence: {confidence:.4f})")

        # Return prediction results
        return jsonify({
            'predicted_class_index': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': {label: float(prob) for label, prob in zip(class_labels, predictions[0])}
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is up and running"""
    return jsonify({'status': 'healthy', 'model': 'lung5.keras'}), 200

@app.route('/')
def home():
    """Home page with basic API instructions"""
    return '''
    <html>
        <head>
            <title>Lung CT Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Lung CT Classification API</h1>
            <p>Upload a lung CT image to analyze for possible conditions.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><code>/predict</code> - POST an image file for classification</li>
                <li><code>/health</code> - GET server health status</li>
            </ul>
            <h2>Supported Classes:</h2>
            <ul>
                <li>adenocarcinoma</li>
                <li>large.cell.carcinoma</li>
                <li>normal</li>
                <li>squamous.cell.carcinoma</li>
            </ul>
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Use 0.0.0.0 to make accessible on network
    app.run(host='0.0.0.0', port=5000, debug=False)