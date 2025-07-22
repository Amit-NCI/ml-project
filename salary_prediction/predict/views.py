from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import joblib
import pandas as pd
import os
import numpy as np
import time
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        if algo_selected == "linear":
            salary_predictor_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
            encoder_path = os.path.join(BASE_DIR, 'encoder.pkl')

            salary_predict_pkl = joblib.load(salary_predictor_path)
            enc_pkl = joblib.load(encoder_path)

            encoded_cat = enc_pkl.transform(cat_df)
            encoded_cat_array = encoded_cat.toarray()
            feature_names = enc_pkl.get_feature_names_out(cat_df.columns)
            encoded_cat_df = pd.DataFrame(encoded_cat_array, columns=feature_names)

            input_df = pd.concat([encoded_cat_df, num_df], axis=1)
            pred = salary_predict_pkl.predict(input_df)
            prediction = round(pred[0], 2)

        elif algo_selected == "xgboost":
            xgb_pipeline_path = os.path.join(BASE_DIR, 'xgboost_salary_model.pkl')
            xgb_pipeline = joblib.load(xgb_pipeline_path)
            input_df = pd.concat([cat_df, num_df], axis=1)
            prediction = round(xgb_pipeline.predict(input_df)[0], 2)

        elif algo_selected == "decision tree":
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

        return render(request, "form.html", {"predicted_salary": prediction})

    except Exception as e:
        return render(request, "form.html", {"error": f"Prediction failed: {e}"})

# def model_metrics(request):
#     algorithm = request.GET.get('algorithm', 'decision tree')

#     try:
#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#         if algorithm == "decision tree":
#             model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
#             X_test_path = os.path.join(BASE_DIR, 'tree_X_test.npy')
#             y_test_path = os.path.join(BASE_DIR, 'tree_y_test.npy')
#             output_json = os.path.join(BASE_DIR, 'metrics_decision_tree_full.json')
#         else:
#             raise ValueError("Only Decision Tree supported in this example.")

#         # Load model and data
#         model = joblib.load(model_path)
#         X_test = np.load(X_test_path, allow_pickle=True)
#         y_test = np.load(y_test_path, allow_pickle=True)

#         # Prediction and timing
#         start = time.time()
#         y_pred = model.predict(X_test)
#         end = time.time()

#         latency = (end - start) / len(X_test)
#         throughput = 1 / latency
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         accuracy = round(r2 * 100, 2)

#         metrics = {
#             "algorithm": "Decision Tree",
#             "metrics": {
#                 "R2_Score": round(r2, 4),
#                 "Accuracy (%)": accuracy,
#                 "MSE": round(mse, 2),
#                 "MAE": round(mae, 2),
#                 "Latency (ms)": round(latency * 1000, 4),
#                 "Throughput (pred/sec)": round(throughput, 2)
#             },
#             "samples": []
#         }

#         # Add top 5 prediction examples
#         for i in range(min(5, len(X_test))):
#             sample = {
#                 "input_features": X_test[i].tolist(),
#                 "actual_salary": y_test[i].item(),
#                 "predicted_salary": round(y_pred[i], 2)
#             }
#             metrics["samples"].append(sample)

#         # Save to JSON file
#         with open(output_json, "w") as f:
#             json.dump(metrics, f, indent=4)

#         return render(request, "metrics.html", {
#             "algorithm": metrics["algorithm"],
#             "latency": metrics["metrics"]["Latency (ms)"],
#             "r2": metrics["metrics"]["R2_Score"],
#             "mse": metrics["metrics"]["MSE"],
#             "mae": metrics["metrics"]["MAE"],
#             "accuracy": metrics["metrics"]["Accuracy (%)"],
#             "throughput": metrics["metrics"]["Throughput (pred/sec)"],
#             "samples": metrics["samples"]
#         })

#     except Exception as e:
#         return render(request, "metrics.html", {"error": str(e)})

def model_metrics(request):
    algorithm = request.GET.get('algorithm', 'decision tree')

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        if algorithm == "decision tree":
            model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
            X_test_path = os.path.join(BASE_DIR, 'tree_X_test.npy')
            y_test_path = os.path.join(BASE_DIR, 'tree_y_test.npy')
            output_json = os.path.join(BASE_DIR, 'metrics_decision_tree_full.json')
            model_name = "Decision Tree"

        elif algorithm == "linear":
            model_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
            X_test_path = os.path.join(BASE_DIR, 'linear_X_test.npy')
            y_test_path = os.path.join(BASE_DIR, 'linear_y_test.npy')
            output_json = os.path.join(BASE_DIR, 'metrics_linear_regression_full.json')
            model_name = "Linear Regression"

        else:
            raise ValueError("Only 'decision tree' and 'linear' algorithms are supported for metrics.")

        # Load model and test data
        model = joblib.load(model_path)
        X_test = np.load(X_test_path, allow_pickle=True)
        y_test = np.load(y_test_path, allow_pickle=True)

        # Prediction and timing
        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()

        latency = (end - start) / len(X_test)
        throughput = 1 / latency
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = round(r2 * 100, 2)

        metrics = {
            "algorithm": model_name,
            "metrics": {
                "R2_Score": round(r2, 4),
                "Accuracy (%)": accuracy,
                "MSE": round(mse, 2),
                "MAE": round(mae, 2),
                "Latency (ms)": round(latency * 1000, 4),
                "Throughput (pred/sec)": round(throughput, 2)
            },
            "samples": []
        }

        for i in range(min(5, len(X_test))):
            sample = {
                "input_features": X_test[i].tolist(),
                "actual_salary": y_test[i].item(),
                "predicted_salary": round(y_pred[i], 2)
            }
            metrics["samples"].append(sample)

        # Save metrics to JSON
        with open(output_json, "w") as f:
            json.dump(metrics, f, indent=4)

        return render(request, "metrics.html", {
            "algorithm": metrics["algorithm"],
            "latency": metrics["metrics"]["Latency (ms)"],
            "r2": metrics["metrics"]["R2_Score"],
            "mse": metrics["metrics"]["MSE"],
            "mae": metrics["metrics"]["MAE"],
            "accuracy": metrics["metrics"]["Accuracy (%)"],
            "throughput": metrics["metrics"]["Throughput (pred/sec)"],
            "samples": metrics["samples"]
        })

    except Exception as e:
        return render(request, "metrics.html", {"error": str(e)})

# from django.shortcuts import render, redirect
# from django.views.decorators.csrf import csrf_exempt
# from django.contrib import messages
# import joblib
# import pandas as pd
# import os
# import numpy as np
# import time
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# @csrf_exempt
# def form(request):
#     return render(request, "form.html")

# def home(request):
#     return render(request, "home.html")

# @csrf_exempt
# def predict_salar_with_linear_regression(request):
#     try:
#         numerical_features = {
#             'remote_ratio': float(request.POST.get('remote_ratio', 0)),
#             'years_experience': float(request.POST.get('years_experience', 0)),
#             'job_description_length': float(request.POST.get('job_description_length', 0)),
#             'benefits_score': float(request.POST.get('benefits_score', 0)),
#         }

#         categorical_features = {
#             'job_title': request.POST.get('job_title', ''),
#             'experience_level': request.POST.get('experience_level', '').strip(),
#             'employment_type': request.POST.get('employment_type', '').strip(),
#             'company_location': request.POST.get('company_location', '').strip(),
#             'company_size': request.POST.get('company_size', '').strip(),
#             'employee_residence': request.POST.get('employee_residence', '').strip(),
#             'education_required': request.POST.get('education_required', '').strip(),
#             'industry': request.POST.get('industry', '').strip(),
#         }

#         cat_df = pd.DataFrame([categorical_features])
#         num_df = pd.DataFrame([numerical_features])

#         algo_selected = request.POST.get("algorithm")

#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#         if algo_selected == "linear":
#             salary_predictor_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
#             encoder_path = os.path.join(BASE_DIR, 'encoder.pkl')

#             salary_predict_pkl = joblib.load(salary_predictor_path)
#             enc_pkl = joblib.load(encoder_path)

#             encoded_cat = enc_pkl.transform(cat_df)
#             encoded_cat_array = encoded_cat.toarray()
#             feature_names = enc_pkl.get_feature_names_out(cat_df.columns)
#             encoded_cat_df = pd.DataFrame(encoded_cat_array, columns=feature_names)

#             input_df = pd.concat([encoded_cat_df, num_df], axis=1)
#             pred = salary_predict_pkl.predict(input_df)
#             prediction = round(pred[0], 2)

#         elif algo_selected == "xgboost":
#             xgb_pipeline_path = os.path.join(BASE_DIR, 'xgboost_salary_model.pkl')
#             xgb_pipeline = joblib.load(xgb_pipeline_path)
#             input_df = pd.concat([cat_df, num_df], axis=1)
#             prediction = round(xgb_pipeline.predict(input_df)[0], 2)

#         elif algo_selected == "decision tree":
#             tree_model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
#             tree_encoder_path = os.path.join(BASE_DIR, 'amit_encoder.pkl')

#             tree_model = joblib.load(tree_model_path)
#             encoders = joblib.load(tree_encoder_path)

#             for col in cat_df.columns:
#                 cat_df[col] = encoders[col].transform(cat_df[col])

#             input_df = pd.concat([cat_df, num_df], axis=1)
#             prediction = round(tree_model.predict(input_df)[0], 2)

#         else:
#             raise ValueError(f"Unsupported algorithm: {algo_selected}")

#         print("Prediction:", prediction)
#         return render(request, "form.html", {"predicted_salary": prediction})

#     except Exception as e:
#         print("Error during prediction:", e)
#         return render(request, "form.html", {"error": f"Prediction failed: {e}"})



# from sklearn.metrics import r2_score

# def model_metrics(request):
#     algorithm = request.GET.get('algorithm', 'linear')

#     try:
#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#         if algorithm == "linear":
#             model_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
#             X_test_path = os.path.join(BASE_DIR, 'linear_X_test.npy')
#             y_test_path = os.path.join(BASE_DIR, 'linear_y_test.npy')
#         elif algorithm == "decision tree":
#             model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
#             X_test_path = os.path.join(BASE_DIR, 'tree_X_test.npy')
#             y_test_path = os.path.join(BASE_DIR, 'tree_y_test.npy')
#         else:
#             raise ValueError("Unsupported algorithm selected.")

#         model = joblib.load(model_path)
#         X_test = np.load(X_test_path, allow_pickle=True)
#         y_test = np.load(y_test_path, allow_pickle=True)

#         # Predictions and metrics
#         start = time.time()
#         y_pred = model.predict(X_test)
#         end = time.time()

#         latency = (end - start) / len(X_test)
#         throughput = 1 / latency
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         accuracy = round(max(0, min(1, r2)) * 100, 2)  # crude approximation

#         context = {
#             'algorithm': algorithm.title(),
#             'latency': round(latency * 1000, 2),
#             'throughput': round(throughput, 2),
#             'mse': round(mse, 2),
#             'mae': round(mae, 2),
#             'r2': round(r2, 4),
#             'accuracy': accuracy
#         }

#         print("Context sent to metrics.html:", context)
#         return render(request, "metrics.html", context)

#     except Exception as e:
#         return render(request, "metrics.html", {"error": str(e)})


# ========================================================================================



# from django.shortcuts import render, redirect
# from django.views.decorators.csrf import csrf_exempt
# from django.contrib import messages

# import joblib
# import pandas as pd
# import os


# @csrf_exempt
# def form(request):
#     return render(request, "form.html")


# def home(request):
#     return render(request, "home.html")


# @csrf_exempt
# def predict_salar_with_linear_regression(request):
#     try:
#         numerical_features = {
#             'remote_ratio': float(request.POST.get('remote_ratio', 0)),
#             'years_experience': float(request.POST.get('years_experience', 0)),
#             'job_description_length': float(request.POST.get('job_description_length', 0)),
#             'benefits_score': float(request.POST.get('benefits_score', 0)),
#         }

#         categorical_features = {
#             'job_title': request.POST.get('job_title', ''),
#             'experience_level': request.POST.get('experience_level', '').strip(),
#             'employment_type': request.POST.get('employment_type', '').strip(),
#             'company_location': request.POST.get('company_location', '').strip(),
#             'company_size': request.POST.get('company_size', '').strip(),
#             'employee_residence': request.POST.get('employee_residence', '').strip(),
#             'education_required': request.POST.get('education_required', '').strip(),
#             'industry': request.POST.get('industry', '').strip(),
#         }

#         cat_df = pd.DataFrame([categorical_features])
#         num_df = pd.DataFrame([numerical_features])

#         algo_selected = request.POST.get("algorithm")

#         if algo_selected == "linear":
#             BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#             salary_predictor_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
#             encoder_path = os.path.join(BASE_DIR, 'encoder.pkl')

#             with open(salary_predictor_path, "rb") as file:
#                 salary_predict_pkl = joblib.load(file)
#             with open(encoder_path, "rb") as file:
#                 enc_pkl = joblib.load(file)

#             encoded_cat = enc_pkl.transform(cat_df)
#             encoded_cat_array = encoded_cat.toarray()
#             feature_names = enc_pkl.get_feature_names_out(cat_df.columns)
#             encoded_cat_df = pd.DataFrame(encoded_cat_array, columns=feature_names)

#             input_df = pd.concat([encoded_cat_df, num_df], axis=1)
#             pred = salary_predict_pkl.predict(input_df)
#             prediction = round(pred[0], 2)

#         elif algo_selected == "xgboost":
#             BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#             xgb_pipeline_path = os.path.join(BASE_DIR, 'xgboost_salary_model.pkl')
#             xgb_pipeline = joblib.load(xgb_pipeline_path)
#             input_df = pd.concat([cat_df, num_df], axis=1)
#             prediction = round(xgb_pipeline.predict(input_df)[0], 2)

#         elif algo_selected == "decision tree":
#             BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#             tree_model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
#             tree_encoder_path = os.path.join(BASE_DIR, 'amit_encoder.pkl')

#             tree_model = joblib.load(tree_model_path)
#             encoders = joblib.load(tree_encoder_path)

#             for col in cat_df.columns:
#                 cat_df[col] = encoders[col].transform(cat_df[col])

#             input_df = pd.concat([cat_df, num_df], axis=1)
#             prediction = round(tree_model.predict(input_df)[0], 2)

#         else:
#             raise ValueError(f"Unsupported algorithm: {algo_selected}")

#         print("Prediction:", prediction)
#         return render(request, "form.html", {"predicted_salary": prediction})

#     except Exception as e:
#         print("Error during prediction:", e)
#         return render(request, "form.html", {"error": f"Prediction failed: {e}"})
# # ==================================================
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import time
# import numpy as np

# # Global storage for metrics
# last_metrics = {}

# def show_metrics(request):
#     return render(request, "metrics.html", context=last_metrics)

# # ===================================================
# import time
# import numpy as np
# from django.shortcuts import render
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# def model_metrics(request):
#     algorithm = request.GET.get('algorithm', 'linear')  # Default to linear

#     try:
#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#         if algorithm == "linear":
#             model_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
#             X_test_path = os.path.join(BASE_DIR, 'linear_X_test.npy')
#             y_test_path = os.path.join(BASE_DIR, 'linear_y_test.npy')
#         elif algorithm == "decision tree":
#             model_path = os.path.join(BASE_DIR, 'amit_model.pkl')
#             X_test_path = os.path.join(BASE_DIR, 'tree_X_test.npy')
#             y_test_path = os.path.join(BASE_DIR, 'tree_y_test.npy')
#         else:
#             raise ValueError("Unsupported algorithm selected.")

#         model = joblib.load(model_path)
#         X_test = np.load(X_test_path, allow_pickle=True)
#         y_test = np.load(y_test_path, allow_pickle=True)

#         # Measure latency
#         start = time.time()
#         y_pred = model.predict(X_test)
#         end = time.time()

#         latency = (end - start) / len(X_test)  # seconds per prediction
#         throughput = 1 / latency  # predictions per second

#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)

#         context = {
#             'algorithm': algorithm,
#             'latency': round(latency * 1000, 2),  # ms
#             'throughput': round(throughput, 2),
#             'mse': round(mse, 2),
#             'mae': round(mae, 2)
#         }
#         print("Context sent to metrics.html:", context)
#         return render(request, "metrics.html", context)


#     except Exception as e:
#         context = {
#             'error': str(e),
#             'algorithm': algorithm
#         }

#     return render(request, "metrics.html", context)
