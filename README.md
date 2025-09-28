# 🧬 Breast Cancer Tissue Classification

A Machine Learning-powered web application for breast tissue classification using a Decision Tree Classifier.
This project demonstrates end-to-end ML model development — from data preprocessing, training, and evaluation to deployment with a user-friendly Streamlit web interface.

# 📌 Project Overview

Breast cancer classification is an essential step in early cancer detection.
This project uses a Decision Tree Classifier trained on a breast tissue dataset to classify tissue samples into multiple classes.

# The project includes:
✅ Data loading & preprocessing
✅ Feature scaling and label encoding
✅ Model training using Decision Tree Classifier
✅ Model evaluation (accuracy, precision, recall, F1-score)
✅ Interactive web app to make real-time predictions

# 🛠 Tech Stack
Language: Python 🐍
Libraries:
pandas, numpy – Data handling
scikit-learn – ML model training & evaluation
streamlit – Web app interface
pickle – Model serialization

# 📊 Dataset
The dataset includes several numerical features such as:
I0
PA500
HFS
DA
Area
A.DA
Max.IP
DR
P

The target column Class represents the type of breast tissue.
Classes were encoded numerically using LabelEncoder before training.

# 📈 Model Training
Algorithm: Decision Tree Classifier (max_depth=8)
Split: 80% training / 20% testing
Preprocessing:
StandardScaler for feature normalization
LabelEncoder for categorical class labels
Performance metrics generated:
Accuracy
Precision / Recall / F1-score
Confusion Matrix

# 🚀 Web App
The Streamlit web app provides a clean, simple interface where users can:
Input feature values
Click Predict
Get instant classification results along with a diagnosis (Positive / Negative)
# 🔗 Example:
streamlit run app.py
Then open the link in your browser (usually http://localhost:8501).

# 📂 Project Structure
├── Breast_Tissue_cancer_detction.py   # Model training, preprocessing & evaluation
├── app.py                             # Streamlit web application
├── breast.pickel                      # Trained Decision Tree model (pickle file)
└── README.md                          # Project documentation

# 🧪 How to Run
Install Dependencies
pip install -r requirements.txt
Run the Streamlit App
streamlit run app.py

# 📊 Sample Output

Model Accuracy: ~81.82%

# Classification Report:
Displays precision, recall, F1-score for each class.

# 🌟 Key Highlights
End-to-end ML workflow
Interactive predictions via Streamlit
Easy-to-deploy & user-friendly UI
High interpretability with Decision Tree

# 🤝 Contributing
Contributions are welcome!
If you’d like to add new models (RandomForest, XGBoost, Neural Networks), open a pull request.

💡 Show your support: ⭐ Star this repo if you find it helpful!
