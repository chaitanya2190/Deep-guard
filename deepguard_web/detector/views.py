import time
import json
import base64
import cv2
import numpy as np

from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.core.files.storage import default_storage

from .forms import VideoUploadForm
from .models import AnalysisResult
from .preprocessing import process_video
from .inference import get_engine


def _encode_image_b64(img_array, apply_colormap=False):
    """Encode a numpy image array to a base64 data URI string."""
    if apply_colormap:
        coloured = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
        _, buf = cv2.imencode('.png', coloured)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def upload_view(request):
    """Main page: upload video and display analysis results."""
    context = {'form': VideoUploadForm()}

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        context['form'] = form

        if form.is_valid():
            video_file = form.cleaned_data['video']
            threshold = form.cleaned_data['threshold']

            # Save uploaded file to media/
            saved_path = default_storage.save(
                f'uploads/{video_file.name}', video_file
            )
            full_path = default_storage.path(saved_path)

            start_time = time.time()

            # Preprocess: extract 8 face frames
            rgb_tensor, dct_tensor, display_crops = process_video(full_path)

            if rgb_tensor is None:
                context['error'] = (
                    'Could not extract 8 face frames from the video. '
                    'Ensure the video contains a clearly visible face and is at least 8 frames long.'
                )
                return render(request, 'detector/upload.html', context)

            # Run inference
            engine = get_engine()
            result = engine.predict(rgb_tensor, dct_tensor, threshold)
            processing_time = round(time.time() - start_time, 2)

            # Encode face crops and heatmaps as base64 for storage and display
            crops_b64 = [_encode_image_b64(crop) for crop in display_crops]
            heatmaps_b64 = [_encode_image_b64(hm, apply_colormap=True) for hm in result['heatmaps']]

            # Overlay heatmaps on face crops for combined visualisation
            overlays_b64 = []
            for crop, hm in zip(display_crops, result['heatmaps']):
                hm_resized = cv2.resize(hm, (crop.shape[1], crop.shape[0]))
                hm_coloured = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(crop_bgr, 0.6, hm_coloured, 0.4, 0)
                _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                overlays_b64.append(base64.b64encode(buf).decode('utf-8'))

            # Save to database
            analysis = AnalysisResult.objects.create(
                video_file=saved_path,
                filename=video_file.name,
                deepfake_probability=result['deepfake_probability'],
                reconstruction_error=result['reconstruction_error'],
                is_anomaly=result['is_anomaly'],
                anomaly_threshold=threshold,
                verdict=result['verdict'],
                explanation=result['explanation'],
                frame_crops_json=json.dumps(crops_b64),
                heatmaps_json=json.dumps(heatmaps_b64),
                processing_time=processing_time,
            )

            context['result'] = analysis
            context['crops_b64'] = crops_b64
            context['heatmaps_b64'] = heatmaps_b64
            context['overlays_b64'] = overlays_b64
            context['frame_data'] = list(zip(crops_b64, heatmaps_b64, overlays_b64))

    return render(request, 'detector/upload.html', context)


def history_view(request):
    """Display paginated list of past analyses."""
    results_list = AnalysisResult.objects.all()
    paginator = Paginator(results_list, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'detector/history.html', {
        'page_obj': page_obj,
    })


def result_detail_view(request, pk):
    """Show detailed result for a single analysis."""
    analysis = get_object_or_404(AnalysisResult, pk=pk)

    crops_b64 = json.loads(analysis.frame_crops_json) if analysis.frame_crops_json else []
    heatmaps_b64 = json.loads(analysis.heatmaps_json) if analysis.heatmaps_json else []
    frame_data = list(zip(crops_b64, heatmaps_b64))

    return render(request, 'detector/result.html', {
        'result': analysis,
        'crops_b64': crops_b64,
        'heatmaps_b64': heatmaps_b64,
        'frame_data': frame_data,
    })
