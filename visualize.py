# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

def visualize_model_performance():
    # 데이터 준비
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 모델 훈련
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 예측 및 혼동 행렬 생성
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 이미지 저장
    plt.savefig('static/images/accuracy_plot.png')

    # 모델과 스케일러 저장
    joblib.dump(model, 'models/iris_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Confusion Matrix saved successfully!")

# 호출
if __name__ == "__main__":
    visualize_model_performance()
