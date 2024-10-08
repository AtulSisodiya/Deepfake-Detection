from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.models import load_model
import os

app = Flask(__name__)
model = load_model('model/deepfake_detector.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the video from the request
    file = request.files['video']
    file_path = 'temp_video.mp4'
    file.save(file_path)  # Save the uploaded video

    # Extract frames from the video
    extract_frames(file_path, 'output/temp_frames', step=30)

    # Load and predict
    frames = []
    for filename in sorted(os.listdir('output/temp_frames')):
        img = cv2.imread(os.path.join('output/temp_frames', filename))
        img = cv2.resize(img, (224, 224))
        frames.append(img)

    frames = np.array(frames)
    predictions = model.predict(frames)

    # Calculate the average prediction
    average_prediction = np.mean(predictions)
    result = "Fake" if average_prediction > 0.5 else "Real"

    # Clean up temporary files
    os.remove(file_path)
    for filename in os.listdir('output/temp_frames'):
        os.remove(os.path.join('output/temp_frames', filename))
    os.rmdir('output/temp_frames')

    return jsonify({'result': result, 'confidence': average_prediction})

def extract_frames(video_path, output_folder, start_frame=0, end_frame=None, step=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
        if frame_count >= start_frame and frame_count % step == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
