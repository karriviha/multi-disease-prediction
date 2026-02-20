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


# ================= BASE DIRECTORY =================
MODELS_DIR = Path(settings.BASE_DIR) / "models"
RESULTS_PATH = MODELS_DIR / "all_results.json"


# ================= LOAD TRAINED MODELS =================
heart_model = joblib.load(MODELS_DIR / "heart_model.pkl")
heart_scaler = joblib.load(MODELS_DIR / "heart_scaler.pkl")

diabetes_model = joblib.load(MODELS_DIR / "diabetes_model.pkl")
diabetes_scaler = joblib.load(MODELS_DIR / "diabetes_scaler.pkl")

cancer_model = joblib.load(MODELS_DIR / "breast_cancer_model.pkl")
cancer_scaler = joblib.load(MODELS_DIR / "breast_cancer_scaler.pkl")

liver_model = joblib.load(MODELS_DIR / "liver_model.pkl")
liver_scaler = joblib.load(MODELS_DIR / "liver_scaler.pkl")


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

@login_required(login_url="login")  
def best_summary(request):

        best_models = []

        if RESULTS_PATH.exists():
            with open(RESULTS_PATH, "r") as f:
                data = json.load(f)

                for disease, models in data.items():
                    best = max(models, key=lambda x: x["F1"])

                    best_models.append({
                        "disease": disease.replace("_", " ").title(),
                        "algorithm": best["Algorithm"],
                        "accuracy": best["Accuracy"],
                        "f1": best["F1"]
                    })

        return render(request, "best_summary.html", {
            "best_models": best_models
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

        for disease_key, models in raw_results.items():

            best_model = max(models, key=lambda x: (x["F1"], x.get("ROC_AUC", 0)))

            formatted_results.append({
                "name": display_names.get(disease_key, disease_key.title()),
                "models": models,
                "best_model": best_model["Algorithm"]
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

            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.fillna(df.median())

            # Drop training target columns
            for col in ["num", "Outcome", "diagnosis", "Stage", "id", "dataset"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            # Select correct model
            if disease == "heart":
                model = heart_model
                scaler = heart_scaler

            elif disease == "diabetes":
                model = diabetes_model
                scaler = diabetes_scaler

            elif disease == "cancer":
                model = cancer_model
                scaler = cancer_scaler

            elif disease == "liver":
                model = liver_model
                scaler = liver_scaler

            else:
                error = "Invalid disease selected."
                return render(request, "all_predict.html", {"error": error})

            X_scaled = scaler.transform(df)
            predictions = model.predict(X_scaled)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_scaled)
                max_prob = np.max(probabilities, axis=1) * 100
            else:
                max_prob = [0] * len(predictions)

            df_original["Prediction"] = predictions
            df_original["Probability (%)"] = np.round(max_prob, 2)

            # Risk Classification
            risk_levels = []

            for pred, prob in zip(predictions, max_prob):

                if pred == 0:
                    risk_levels.append("No Risk")
                else:
                    if prob >= 85:
                        risk_levels.append("High Risk")
                    elif prob >= 65:
                        risk_levels.append("Moderate Risk")
                    else:
                        risk_levels.append("Mild Risk")

            df_original["Risk Level"] = risk_levels

            # Summary Card Data
            total_records = len(df_original)

            high_count = risk_levels.count("High Risk")
            moderate_count = risk_levels.count("Moderate Risk")
            mild_count = risk_levels.count("Mild Risk")
            none_count = risk_levels.count("No Risk")

            # Calculate percentages safely
            if total_records > 0:
                high_percent = round((high_count / total_records) * 100, 2)
                moderate_percent = round((moderate_count / total_records) * 100, 2)
                mild_percent = round((mild_count / total_records) * 100, 2)
                none_percent = round((none_count / total_records) * 100, 2)
            else:
                high_percent = moderate_percent = mild_percent = none_percent = 0

            summary = {
                "total": total_records,
                "high": high_count,
                "moderate": moderate_count,
                "mild": mild_count,
                "none": none_count,
                "high_percent": high_percent,
                "moderate_percent": moderate_percent,
                "mild_percent": mild_percent,
                "none_percent": none_percent,
            }
            records = df_original.to_dict(orient="records")


        except Exception as e:
            error = str(e)

    return render(request, "all_predict.html", {
        "records": records,
        "error": error,
        "summary": summary
    })
