"""
REST API endpoint for DeepGuard.
POST /api/detect/ — upload video, get JSON detection result.
"""

import time
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage

from .models import AnalysisResult
from .preprocessing import process_video
from .inference import get_engine


@csrf_exempt
@require_POST
def detect_api(request):
    """
    API endpoint for deepfake detection.

    POST /api/detect/
    Body: multipart/form-data
        - video: video file (required)
        - threshold: float 0.01-0.50 (optional, default 0.15)

    Returns: JSON
        {
            "id": 1,
            "filename": "test.mp4",
            "deepfake_probability": 87.32,
            "reconstruction_error": 0.2341,
            "is_anomaly": true,
            "verdict": "MANIPULATED",
            "explanation": "...",
            "processing_time_seconds": 3.42,
            "created_at": "2026-07-25T15:00:00Z"
        }
    """
    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video file provided. Send as multipart form field "video".'}, status=400)

    # Validate extension
    ext = video_file.name.rsplit('.', 1)[-1].lower()
    if ext not in ('mp4', 'avi', 'mov', 'mkv'):
        return JsonResponse({'error': f'Unsupported format: .{ext}. Use .mp4, .avi, .mov, or .mkv'}, status=400)

    # Validate size
    if video_file.size > 100 * 1024 * 1024:
        return JsonResponse({'error': 'File too large. Maximum 100 MB.'}, status=400)

    # Parse threshold
    try:
        threshold = float(request.POST.get('threshold', 0.15))
        threshold = max(0.01, min(0.50, threshold))
    except (ValueError, TypeError):
        threshold = 0.15

    # Save file
    saved_path = default_storage.save(f'uploads/{video_file.name}', video_file)
    full_path = default_storage.path(saved_path)

    start_time = time.time()

    # Preprocess
    rgb_tensor, dct_tensor, display_crops = process_video(full_path)
    if rgb_tensor is None:
        return JsonResponse({
            'error': 'Could not extract 8 face frames. Ensure video contains a visible face and is at least 8 frames.'
        }, status=422)

    # Inference
    engine = get_engine()
    result = engine.predict(rgb_tensor, dct_tensor, threshold)
    processing_time = round(time.time() - start_time, 2)

    # Persist
    analysis = AnalysisResult.objects.create(
        video_file=saved_path,
        filename=video_file.name,
        deepfake_probability=result['deepfake_probability'],
        reconstruction_error=result['reconstruction_error'],
        is_anomaly=result['is_anomaly'],
        anomaly_threshold=threshold,
        verdict=result['verdict'],
        explanation=result['explanation'],
        processing_time=processing_time,
    )

    return JsonResponse({
        'id': analysis.pk,
        'filename': analysis.filename,
        'deepfake_probability': analysis.deepfake_probability,
        'reconstruction_error': analysis.reconstruction_error,
        'is_anomaly': analysis.is_anomaly,
        'verdict': analysis.verdict,
        'explanation': analysis.explanation,
        'processing_time_seconds': processing_time,
        'created_at': analysis.created_at.isoformat(),
    })
