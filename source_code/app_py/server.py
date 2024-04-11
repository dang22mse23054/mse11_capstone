import os
import json
import time
import numpy as np
from services.model_service import Constants, ModelService

from flask import Flask, request, jsonify, make_response, g

UPLOAD_FOLDER = '/tmp'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Server is running')

model_service = ModelService()

@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.data)
        print(request.files)
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
                    
                    print(ag_output)
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
            
            result = {
                # random choice when the max value is the same (ex: [1,3,3,4,2] => index 1 or 2 should be choiced randomly)
                'age_groups': int(np.random.choice(np.where(age_groups == age_groups.max())[0])), 
                'genders': int(np.argmax(genders)),
                'faces': output_faces
            } if len(output_faces) > 0 else { 'faces': output_faces}
            print(result)
            return jsonify(result)
        
        return 'Image file is None'
    
    except Exception as error:
        app.logger.error(error)

if __name__ == "__main__":
    host = os.environ.get('APP_HOST', '0.0.0.0')
    port = os.environ.get('APP_PORT', 5001)
    app.run(debug=True, host=host, port=port)