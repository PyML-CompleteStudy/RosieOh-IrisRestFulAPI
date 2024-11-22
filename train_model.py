# model_training.py
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

# 모델과 스케일러 저장
joblib.dump(model, 'models/iris_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model and scaler saved successfully!")
