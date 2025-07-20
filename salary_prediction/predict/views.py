from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

import joblib
import pandas as pd
import os


@csrf_exempt
def form(request):
    return render(request, "form.html")


def home(request):
    return render(request, "home.html")


@csrf_exempt
def predict_salar_with_linear_regression(request):
    try:
        numerical_features = {
            'remote_ratio': float(request.POST.get('remote_ratio', 0)),
            'years_experience': float(request.POST.get('years_experience', 0)),
            'job_description_length': float(request.POST.get('job_description_length', 0)),
            'benefits_score': float(request.POST.get('benefits_score', 0)),
        }

        categorical_features = {
            'job_title': request.POST.get('job_title', ''),
            'experience_level': request.POST.get('experience_level', '').strip(),
            'employment_type': request.POST.get('employment_type', '').strip(),
            'company_location': request.POST.get('company_location', '').strip(),
            'company_size': request.POST.get('company_size', '').strip(),
            'employee_residence': request.POST.get('employee_residence', '').strip(),
            'education_required': request.POST.get('education_required', '').strip(),
            'industry': request.POST.get('industry', '').strip(),
        }

        cat_df = pd.DataFrame([categorical_features])
        num_df = pd.DataFrame([numerical_features])

        algo_selected = request.POST.get("algorithm")

        if algo_selected == "linear":
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            salary_predictor_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
            encoder_path = os.path.join(BASE_DIR, 'encoder.pkl')

            with open(salary_predictor_path, "rb") as file:
                salary_predict_pkl = joblib.load(file)
            with open(encoder_path, "rb") as file:
                enc_pkl = joblib.load(file)

            encoded_cat = enc_pkl.transform(cat_df)
            encoded_cat_array = encoded_cat.toarray()
            feature_names = enc_pkl.get_feature_names_out(cat_df.columns)
            encoded_cat_df = pd.DataFrame(encoded_cat_array, columns=feature_names)

            input_df = pd.concat([encoded_cat_df, num_df], axis=1)
            pred = salary_predict_pkl.predict(input_df)
            prediction = round(pred[0], 2)

        elif algo_selected == "xgboost":
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            xgb_pipeline_path = os.path.join(BASE_DIR, 'xgboost_salary_model.pkl')
            xgb_pipeline = joblib.load(xgb_pipeline_path)
            input_df = pd.concat([cat_df, num_df], axis=1)
            prediction = round(xgb_pipeline.predict(input_df)[0], 2)

        elif algo_selected == "decision tree":
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            tree_model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
            tree_encoder_path = os.path.join(BASE_DIR, 'amit_encoder.pkl')

            tree_model = joblib.load(tree_model_path)
            encoders = joblib.load(tree_encoder_path)

            for col in cat_df.columns:
                cat_df[col] = encoders[col].transform(cat_df[col])

            input_df = pd.concat([cat_df, num_df], axis=1)
            prediction = round(tree_model.predict(input_df)[0], 2)

        else:
            raise ValueError(f"Unsupported algorithm: {algo_selected}")

        print("Prediction:", prediction)
        return render(request, "form.html", {"predicted_salary": prediction})

    except Exception as e:
        print("Error during prediction:", e)
        return render(request, "form.html", {"error": f"Prediction failed: {e}"})
