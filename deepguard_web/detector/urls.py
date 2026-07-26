from django.urls import path
from . import views, api_views

app_name = 'detector'

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('history/', views.history_view, name='history'),
    path('result/<int:pk>/', views.result_detail_view, name='result_detail'),
    path('api/detect/', api_views.detect_api, name='api_detect'),
]
