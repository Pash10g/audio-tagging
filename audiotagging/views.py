import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from panns_inference import AudioTagging
import librosa
import numpy as np
import os

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def get_embedding(audio_file):
    model = AudioTagging(checkpoint_path=None, device='cpu')
    a, _ = librosa.load(audio_file, sr=44100)
    query_audio = a[None, :]
    _, emb = model.inference(query_audio)
    return normalize(emb[0])

@csrf_exempt
def audio_tagging_view(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        print(f"Received file: {audio_file.name}, size: {audio_file.size}")  # Debugging line
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            for chunk in audio_file.chunks():
                temp_audio_file.write(chunk)
            temp_audio_file.flush()  # Ensure all data is written to disk before proceeding
            print(f"Temp file: {temp_audio_file.name}, size: {os.path.getsize(temp_audio_file.name)}")  # Debugging line
        embedding = get_embedding(temp_audio_file.name)
        return JsonResponse({'embedding': embedding.tolist()})
    return JsonResponse({'error': 'Invalid request'}, status=400)
