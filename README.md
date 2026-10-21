# Heart Disease Prediction System

A machine learning web app that predicts whether a patient is at higher or lower risk of heart disease from four inputs. Built for AIC354 Machine Learning Fundamentals (Assignment 2) at COMSATS University Islamabad, Lahore Campus.

**Live demo:** https://ml-heart-disease-prediction-model-by-wajahat.streamlit.app/

## Dataset
- [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease), 303 patients
- Target: `0` = no disease, `1` = disease (original levels 1-4 merged into 1)

## Input features
| Feature | Meaning |
|---|---|
| `cp` | Chest pain type (label encoded) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina, yes/no (label encoded) |
| `oldpeak` | ST depression induced by exercise |

## Project steps
1. Data collection
2. Data understanding
3. Preprocessing (4 features selected, missing values dropped, binary target)
4. Label encoding
5. Model selection and training (Logistic Regression, Random Forest)
6. Evaluation (accuracy, classification report, confusion matrix)
7. Deployment on Streamlit Community Cloud

## Files
- `heart_disease_prediction.ipynb`: full notebook
- `app.py`: Streamlit web app
- `heart_model.pkl`: trained model pipeline
- `heart.csv`: dataset
- `requirements.txt`: dependencies

## Run locally
```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Results
- Best model: _fill in_
- Test accuracy: _fill in_

## Disclaimer
This is a student project and not a medical tool. See a doctor for real health concerns.

## Author
Wajahat · CUI Lahore<img width="739" height="307" alt="image" src="https://github.com/user-attachments/assets/792f0252-1938-477f-a1a4-32f0dd3df20d" />
