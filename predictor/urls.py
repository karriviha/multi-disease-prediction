from django.urls import path
from . import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("register/", views.register, name="register"),
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),

    path("dashboard/", views.dashboard, name="dashboard"),

    path("compare/", views.comparison_graph, name="comparison"),
    path("detect/", views.all_predict, name="all_predict"),
    path("best-summary/", views.best_summary, name="best_summary"),
]

