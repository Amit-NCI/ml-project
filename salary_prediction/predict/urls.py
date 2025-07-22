# # from django.contrib import admin
# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home'),
#     path('form', views.form, name='form'),
#     path('lr', views.predict_salar_with_linear_regression, name='predict_salary'),
#     path('metrics/', views.show_metrics, name='show_metrics'),
# ]
# # 


from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('form', views.form, name='form'),
    path('lr', views.predict_salar_with_linear_regression, name='predict_salary'),
    path('metrics', views.model_metrics, name='show_metrics'),
]
