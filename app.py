from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

# Flask 앱 생성
app = Flask(__name__)

# 모델 로드
model = joblib.load("models/iris_model.pkl")

# JWT 설정
app.config["JWT_SECRET_KEY"] = "your-secret-key"
jwt = JWTManager(app)

# 로그 설정
logging.basicConfig(filename='api.log', level=logging.INFO)

# 헬스체크 엔드포인트
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

# 예측 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # 입력값 검증
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # 입력값 가져오기
        features = np.array([[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]])

        # 예측
        prediction = model.predict(features)[0]
        class_names = ["Setosa", "Versicolor", "Virginica"]
        result = {"prediction": class_names[int(prediction)]}

        # 로그 저장
        logging.info(f"{datetime.now()} - Input: {data}, Prediction: {result}")
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 모델 정보 엔드포인트
@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "accuracy": 0.95,  # 학습 시 측정한 값
        "algorithm": "Random Forest",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    })

# 배치 예측 엔드포인트
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        features = np.array(data["samples"])
        predictions = model.predict(features)
        class_names = ["Setosa", "Versicolor", "Virginica"]
        result = [class_names[int(pred)] for pred in predictions]
        return jsonify({"predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 시각화 엔드포인트
@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = request.json
        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]
        plt.figure()
        plt.bar(["sepal_length", "sepal_width", "petal_length", "petal_width"], features)
        plt.title("Iris Features")
        plt.xlabel("Features")
        plt.ylabel("Values")

        # 이미지 저장 및 반환
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return jsonify({"image": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 로그인 엔드포인트 (JWT)
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    # 간단한 인증 (실제 프로젝트에서는 DB 확인 필요)
    if username == "admin" and password == "password":
        token = create_access_token(identity=username)
        return jsonify(access_token=token)
    return jsonify({"error": "Invalid credentials"}), 401

# 보안 예측 엔드포인트 (JWT 필요)
@app.route('/secure-predict', methods=['POST'])
@jwt_required()
def secure_predict():
    try:
        data = request.json
        features = np.array([[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]])
        prediction = model.predict(features)[0]
        class_names = ["Setosa", "Versicolor", "Virginica"]
        return jsonify({"prediction": class_names[int(prediction)]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True)
