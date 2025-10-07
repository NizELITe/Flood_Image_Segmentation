# from flask import Flask, render_template, request, redirect, url_for
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import cv2

# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# from model_file import EncoderBlock, DecoderBlock, AttentionGate

# app = Flask(__name__)

# # # Paths
# # UPLOAD_FOLDER = "static/uploads"
# # RESULT_FOLDER = "static/results"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(RESULT_FOLDER, exist_ok=True)



# # Configure upload and result folders
# app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
# app.config['RESULT_FOLDER'] = os.path.join('static', 'results')

# # Make sure folders exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# # Load your trained model
# # MODEL_PATH = "best_model.h5"
# # model = load_model(MODEL_PATH)



# model = load_model(
#     "flood_segmentation_model.h5",
#     custom_objects={
#         "EncoderBlock": EncoderBlock,
#         "DecoderBlock": DecoderBlock,
#         "AttentionGate": AttentionGate
#     }
# )


# # Image size used during training
# IMG_HEIGHT, IMG_WIDTH = 256, 256

# def preprocess_image(image_path):
#     """Load and preprocess image for model"""
#     img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
#     img_array = img_to_array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# def postprocess_mask(pred_mask, save_path, original_path):
#     """Save predicted mask overlay on original image"""
#     pred_mask = (pred_mask > 0.5).astype("uint8")[0]  # binary mask

#     # Resize mask back to original
#     orig = cv2.imread(original_path)
#     mask_resized = cv2.resize(pred_mask, (orig.shape[1], orig.shape[0]))

#     # Apply mask overlay (green)
#     overlay = orig.copy()
#     overlay[mask_resized == 1] = [0, 255, 0]  

#     result = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
#     cv2.imwrite(save_path, result)

# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     if request.method == "POST":
# #         if "file" not in request.files:
# #             return redirect(request.url)
# #         file = request.files["file"]
# #         if file.filename == "":
# #             return redirect(request.url)

# #         # Save uploaded image
# #         img_path = os.path.join(UPLOAD_FOLDER, file.filename)
# #         file.save(img_path)

# #         # Preprocess + Predict
# #         img_array = preprocess_image(img_path)
# #         pred_mask = model.predict(img_array)

# #         # Save result
# #         result_path = os.path.join(RESULT_FOLDER, "result_" + file.filename)
# #         postprocess_mask(pred_mask, result_path, img_path)
        

# #         return render_template("index.html",
# #                                uploaded_image=img_path,
# #                                result_image=result_path)

# #     return render_template("index.html")

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['file']
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         # Read and preprocess image
#         img = cv2.imread(filepath)
#         img_resized = cv2.resize(img, (256, 256))
#         img_array = np.expand_dims(img_resized, axis=0) / 255.0

#         # Predict mask
#         prediction = model.predict(img_array)[0]

#         # Convert prediction -> binary mask
#         pred_mask = (prediction > 0.5).astype(np.uint8) * 255  

#         # Resize mask back to original image size
#         pred_img = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

#         # Save result
#         result_filename = f"result_{file.filename}"
#         result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
#         cv2.imwrite(result_path, pred_img)

#         return render_template(
#             "index.html",
#             uploaded_filename=file.filename,
#             result_filename="result_" + file.filename
#         )

#     return render_template('index.html')

       

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

from werkzeug.utils import secure_filename
from model_file import EncoderBlock, DecoderBlock, AttentionGate

app = Flask(__name__)

# Configure upload and result folders
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')

# Make sure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load model with custom layers
model = load_model(
    "flood_segmentation_model.h5",
    custom_objects={
        "EncoderBlock": EncoderBlock,
        "DecoderBlock": DecoderBlock,
        "AttentionGate": AttentionGate
    }
)

# Image size used during training
IMG_HEIGHT, IMG_WIDTH = 256, 256

def preprocess_image(image_path):
    """Load and preprocess image for model"""
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)

#         # Save uploaded image
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         # Preprocess
#         img_array = preprocess_image(filepath)

#         # Predict mask
#         prediction = model.predict(img_array)[0]

#         # Convert prediction -> binary mask
#         pred_mask = (prediction > 0.5).astype(np.uint8) * 255  

#         # Read original image
#         orig = cv2.imread(filepath)

#         # Resize mask to original size
#         mask_resized = cv2.resize(pred_mask, (orig.shape[1], orig.shape[0]))

#         # --- Save raw mask ---
#         result_filename = f"result_{file.filename}"
#         result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
#         cv2.imwrite(result_path, mask_resized)

#         # --- Create overlay (highlight flooded area in red) ---
#         overlay = orig.copy()
#         overlay[mask_resized == 255] = [0, 0, 255]  # red for flooded
#         blended = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)

#         overlay_filename = f"overlay_{file.filename}"
#         overlay_path = os.path.join(app.config['RESULT_FOLDER'], overlay_filename)
#         cv2.imwrite(overlay_path, blended)

#         return render_template(
#             "index.html",
#             uploaded_filename=file.filename,
#             result_filename=result_filename,
#             overlay_filename=overlay_filename
#         )

#     return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read and preprocess image
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (256, 256))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Predict mask
        prediction = model.predict(img_array)[0]

        # Binary mask
        pred_mask = (prediction > 0.5).astype(np.uint8) * 255  

        # Resize mask back to original size
        mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

        # Save segmented mask
        mask_filename = f"mask_{file.filename}"
        mask_path = os.path.join(app.config['RESULT_FOLDER'], mask_filename)
        cv2.imwrite(mask_path, mask_resized)

        # Create overlayed image (green mask on original)
        overlay = img.copy()
        overlay[mask_resized == 255] = [0, 255, 0]  
        overlay_result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        overlay_filename = f"overlay_{file.filename}"
        overlay_path = os.path.join(app.config['RESULT_FOLDER'], overlay_filename)
        cv2.imwrite(overlay_path, overlay_result)

        # return render_template(
        #     "index.html",
        #     uploaded_image=filepath,
        #     mask_image=mask_path,
        #     overlay_image=overlay_path
        # )
        return render_template(
        "index.html",
        uploaded_image=f"uploads/{file.filename}",
        mask_image=f"results/mask_{file.filename}",
        overlay_image=f"results/overlay_{file.filename}"
        )


    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
