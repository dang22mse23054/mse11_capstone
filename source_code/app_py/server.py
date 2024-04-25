import os, json, cv2, time
import traceback
import numpy as np
from services.model_service import Constants, ModelService
from services.attention_service import AttentionService
from utils.decoder import NumpyEncoder

from flask import Flask, request, jsonify, make_response, g

UPLOAD_FOLDER = '/tmp'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Server is running')

model_service = ModelService()
POSITION = Constants.Position()
attention_service = AttentionService()

AGE = Constants.Age()
AGES = list(AGE.Groups.keys())

@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # print(request.data)
        # print(request.files)
        if 'img_file' not in request.files:
            return 'Missing image file in the request'
        
        img_file = request.files['img_file']
        if img_file is not None:
            
            output_faces = model_service.predict_for(Constants.Models.FACE_MODEL, img_file)
            
            # 5 Age groups
            age_groups = np.zeros(5)
            genders = np.zeros(2)
            if (len(output_faces) > 0):
                for face_info in output_faces:
                    face_img = face_info['face_img']
                    ag_output = model_service.predict_for(Constants.Models.AGE_GENDER_MODEL, face_img)

                    if (ag_output is not None):
                        age_group = ag_output['age_group']
                        face_info['age'] = age_group
                        if (age_group is not None and age_group >= 0 and age_group < 5):
                            age_groups[age_group] += 1
                            
                        gender = ag_output['gender']
                        face_info['gender'] = gender
                        if (gender is not None):
                            genders[gender] += 1
                    
                    # Remove unnecessary data
                    del face_info['face_img']
            
            print(f'> age_groups = {age_groups}')
            print(f'> genders = {genders}')
            result = {
                # random choice when the max value is the same (ex: [1,3,3,4,2] => index 1 or 2 should be choiced randomly)
                'majority_age': int(np.random.choice(np.where(age_groups == age_groups.max())[0])), 
                'majority_gender': int(np.argmax(genders)),
                'faces': output_faces
            } if len(output_faces) > 0 else { 'faces': output_faces}

            print('----------- Predict result -----------')
            if len(output_faces) > 0:
                print({
                    'majority_age': AGES[result['majority_age']],
                    'majority_gender': 'Female' if result['majority_age'] else 'Male',
                })
            else:
                print('No face detected')
            print('--------------------------------------')
    
            return jsonify(result)
        
        return 'Image file is None'
    
    except Exception as error:
        app.logger.error(error)
        print(traceback.format_exc())


@app.route("/log/<video_id>", methods=['POST'])
def log_ads(video_id):
    try:
        # print(request.data)
        # print(request.files)
        if 'img_file' not in request.files:
            return 'Missing image file in the request'
        
        img_file = request.files['img_file']
        
        if img_file is not None:
            # Face detection
            output_faces = model_service.predict_for(Constants.Models.FACE_MODEL, img_file)
            print(output_faces)
            log_item = {
                'videoId': int(video_id),
                'createdAt': int(time.time() * 1000),
                'gender': np.zeros(2),
                'age': np.zeros(5),
                'happy': [
                    # gender
                     np.zeros(2),
                    # age
                     np.zeros(5),
                ]
            }
            people = []
            if (len(output_faces) > 0):
                i = 0
                for face_info in output_faces:
                    i += 1
                    # Prepare data
                    person_info = {
                        'age': None,
                        'gender': None,
                        'is_attention': None,
                        'emotion': None,
                    }
                    face_img = face_info['face_img']

                    # Check attention
                    face_img = np.array(face_img, dtype=np.uint8)
                    att_output = attention_service.predict(face_img)
                    print(att_output)

                    # Skip if NO attetion
                    if (att_output is None or len(att_output) == 0 or att_output[0]['is_attention'] is not True):
                        continue
                    
                    predict = att_output[0]
                    person_info['is_attention'] = predict['is_attention']

                    # Check Age/Gender
                    ag_output = model_service.predict_for(Constants.Models.AGE_GENDER_MODEL, face_img)
                    
                    if (ag_output is not None):
                        age_group = ag_output['age_group']
                        person_info['age'] = age_group
                        
                        gender = ag_output['gender']
                        person_info['gender'] = gender

                        # increase counter
                        log_item['age'][age_group] += 1
                        log_item['gender'][gender] += 1
                        
                    # Check Emotion
                    e_output = model_service.predict_for(Constants.Models.EMOTION_MODEL, face_img)
                    if (e_output is not None):
                        e_index = e_output['idx']
                        person_info['emotion'] = e_index
                        # increase counter
                        if (Constants.Emotion.Groups[e_index] == Constants.Emotion.HAPPY):
                              log_item['happy'][0][gender] += 1
                              log_item['happy'][1][age_group] += 1

                    # Add to list
                    people.append(person_info)

            print(people)
            log_item = json.dumps(log_item, cls=NumpyEncoder)
            print(log_item)
            return log_item
        
        return 'Image file is None'
    
    except Exception as error:
        app.logger.error(error)
        print(traceback.format_exc())


if __name__ == "__main__":
    host = os.environ.get('APP_HOST', '0.0.0.0')
    port = os.environ.get('APP_PORT', 5001)
    app.run(debug=True, host=host, port=port)