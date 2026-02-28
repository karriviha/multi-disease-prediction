import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.conf import settings

from .models import UserProfile


# ================= PATHS =================
MODELS_DIR = Path(settings.BASE_DIR) / "models"
RESULTS_PATH = MODELS_DIR / "all_results.json"


# ================= LOAD PIPELINE MODELS =================
heart_model = joblib.load(MODELS_DIR / "heart_model.pkl")
diabetes_model = joblib.load(MODELS_DIR / "diabetes_model.pkl")
cancer_model = joblib.load(MODELS_DIR / "breast_cancer_model.pkl")
liver_model = joblib.load(MODELS_DIR / "liver_model.pkl")


# ================= AUTH =================

def landing(request):
    return render(request, "landing.html")


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        email = request.POST["email"]
        mobile = request.POST["mobile"]
        address = request.POST["address"]

        if User.objects.filter(username=username).exists():
            return render(request, "register.html", {"error": "Username already exists"})

        user = User.objects.create_user(username=username, password=password, email=email)
        UserProfile.objects.create(user=user, mobile=mobile, address=address)

        return redirect("login")

    return render(request, "register.html")


def user_login(request):
    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST["username"],
            password=request.POST["password"]
        )

        if user:
            login(request, user)
            return redirect("dashboard")
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})

    return render(request, "login.html")


def user_logout(request):
    logout(request)
    return redirect("login")


# ================= DASHBOARD =================

@login_required(login_url="login")
def dashboard(request):
    return render(request, "dashboard.html")


# ================= BEST SUMMARY =================

@login_required(login_url="login")
def best_summary(request):

    best_models = []

    try:
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)

        for disease_key, ablation_data in data.items():

            all_models = []

            for smote_label, models in ablation_data.items():

                if not isinstance(models, list):
                    continue

                for model in models:
                    if not isinstance(model, dict):
                        continue

                    model_copy = {**model}
                    model_copy["SMOTE"] = smote_label
                    all_models.append(model_copy)

            if not all_models:
                continue

            best = max(all_models, key=lambda x: x.get("F1_mean", 0))

            best_details = ablation_data.get("Best_Model_Details", {})
            conf_matrix = best_details.get("Confusion_Matrix", [])

            best_models.append({
                "disease": disease_key.replace("_", " ").title(),
                "algorithm": best["Algorithm"],
                "accuracy": best["Accuracy_mean"],
                "f1": best["F1_mean"],
                "smote": best["SMOTE"],
                "confusion_matrix": conf_matrix
            })

        return render(request, "best_summary.html", {
            "best_models": best_models
        })

    except Exception as e:
        return render(request, "best_summary.html", {
            "error": str(e)
        })

# ================= COMPARISON PAGE =================
@login_required(login_url="login")
def comparison_graph(request):

    try:
        with open(RESULTS_PATH, "r") as f:
            raw_results = json.load(f)

        formatted_results = []

        display_names = {
            "heart": "Heart",
            "diabetes": "Diabetes",
            "breast_cancer": "Breast Cancer",
            "liver": "Liver"
        }

        for disease_key, ablation_data in raw_results.items():

            all_models = []
            without_smote_models = []
            with_smote_models = []

            for smote_label, models in ablation_data.items():

                if not isinstance(models, list):
                    continue

                for model in models:

                    if not isinstance(model, dict):
                        continue

                    model_copy = {**model}
                    model_copy["SMOTE"] = smote_label
                    all_models.append(model_copy)

                    if smote_label == "Without_SMOTE":
                        without_smote_models.append(model_copy)

                    elif smote_label == "With_SMOTE":
                        with_smote_models.append(model_copy)

            if not all_models:
                continue

            overall_best = max(all_models, key=lambda x: x.get("F1_mean", 0))

            best_without = (
                max(without_smote_models, key=lambda x: x.get("F1_mean", 0))
                if without_smote_models else None
            )

            best_with = (
                max(with_smote_models, key=lambda x: x.get("F1_mean", 0))
                if with_smote_models else None
            )

            formatted_results.append({
                "name": display_names.get(disease_key, disease_key.title()),
                "models": all_models,
                "overall_best": overall_best["Algorithm"],
                "overall_best_smote": overall_best["SMOTE"],
                "best_without": best_without["Algorithm"] if best_without else None,
                "best_with": best_with["Algorithm"] if best_with else None,
            })

        return render(request, "comparison.html", {
            "results": formatted_results
        })

    except Exception as e:
        return render(request, "comparison.html", {
            "error": str(e)
        })


# ================= PREDICTION PAGE =================

@login_required(login_url="login")
def all_predict(request):

    records = None
    error = None
    summary = None

    if request.method == "POST":

        disease = request.POST.get("disease")

        if "csv_file" not in request.FILES:
            error = "Please upload a CSV file."
            return render(request, "all_predict.html", {"error": error})

        try:
            df = pd.read_csv(request.FILES["csv_file"])
            df_original = df.copy()

            # ================= CLEAN INPUT =================
            df = df.apply(pd.to_numeric, errors="coerce")

            # Replace infinite values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Fill ALL NaN values safely
            df.fillna(0, inplace=True)

            # ================= SELECT MODEL =================
            if disease == "heart":
                model = heart_model
                df.drop(columns=["num", "id", "dataset"], errors="ignore", inplace=True)

            elif disease == "diabetes":
                model = diabetes_model
                df.drop(columns=["Outcome"], errors="ignore", inplace=True)

            elif disease == "cancer":
                model = cancer_model
                df.drop(columns=["diagnosis", "id", "Unnamed: 32"], errors="ignore", inplace=True)

            elif disease == "liver":
                model = liver_model
                df.drop(columns=["Stage"], errors="ignore", inplace=True)

            else:
                error = "Invalid disease selected."
                return render(request, "all_predict.html", {"error": error})

            # ================= ALIGN COLUMNS SAFELY =================
            if hasattr(model, "feature_names_in_"):
                df = df[model.feature_names_in_]

            # ================= PREDICTION =================
            predictions = model.predict(df)

            # 🔥 FIX FOR LIVER (convert back to original stage)
            if disease == "liver":
                predictions = predictions + 1

            # ================= PROBABILITY =================
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(df)
                max_prob = np.max(probabilities, axis=1) * 100
            else:
                max_prob = [0] * len(predictions)

            # ================= STORE RESULTS =================
            df_original["Prediction"] = predictions
            df_original["Probability (%)"] = [f"{p:.2f}%" for p in max_prob]

            # ================= RISK CLASSIFICATION =================
            risk_levels = []

            for pred, prob in zip(predictions, max_prob):

                    if disease in ["heart", "diabetes", "cancer"]:

                        if pred == 0:
                            risk_levels.append("No Risk")
                        else:
                            if prob >= 85:
                                risk_levels.append("High Risk")
                            elif prob >= 65:
                                risk_levels.append("Moderate Risk")
                            else:
                                risk_levels.append("Mild Risk")

                    elif disease == "liver":

                        if pred == 1:
                            risk_levels.append("Mild Risk")
                        elif pred == 2:
                            risk_levels.append("Moderate Risk")
                        elif pred == 3:
                            risk_levels.append("High Risk")
                        elif pred == 4:
                            risk_levels.append("Very High Risk")

            df_original["Risk Level"] = risk_levels

            # ================= SUMMARY =================
            # ================= SUMMARY =================
            total_records = len(df_original)

            if disease == "liver":

                high_count = risk_levels.count("High Risk") + risk_levels.count("Very High Risk")
                moderate_count = risk_levels.count("Moderate Risk")
                mild_count = risk_levels.count("Mild Risk")
                none_count = 0

                summary = {
                    "total": total_records,
                    "high": high_count,
                    "moderate": moderate_count,
                    "mild": mild_count,
                    "none": none_count,
                    "high_percent": round((high_count / total_records) * 100, 2) if total_records else 0,
                    "moderate_percent": round((moderate_count / total_records) * 100, 2) if total_records else 0,
                    "mild_percent": round((mild_count / total_records) * 100, 2) if total_records else 0,
                    "none_percent": 0,
                }

            else:
                high_count = risk_levels.count("High Risk")
                moderate_count = risk_levels.count("Moderate Risk")
                mild_count = risk_levels.count("Mild Risk")
                none_count = risk_levels.count("No Risk")

                summary = {
                    "total": total_records,
                    "high": high_count,
                    "moderate": moderate_count,
                    "mild": mild_count,
                    "none": none_count,
                    "high_percent": round((high_count / total_records) * 100, 2) if total_records else 0,
                    "moderate_percent": round((moderate_count / total_records) * 100, 2) if total_records else 0,
                    "mild_percent": round((mild_count / total_records) * 100, 2) if total_records else 0,
                    "none_percent": round((none_count / total_records) * 100, 2) if total_records else 0,
                }

            records = df_original.to_dict(orient="records")

        except Exception as e:
            error = str(e)

    return render(request, "all_predict.html", {
        "records": records,
        "error": error,
        "summary": summary
    })