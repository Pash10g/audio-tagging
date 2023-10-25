from django.urls import path
from audiotagging import views

urlpatterns = [
    path('audio-tagging/', views.audio_tagging_view, name='audio_tagging'),
]

