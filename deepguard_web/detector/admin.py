from django.contrib import admin
from .models import AnalysisResult


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['filename', 'verdict', 'deepfake_probability', 'reconstruction_error', 'processing_time', 'created_at']
    list_filter = ['verdict', 'is_anomaly', 'created_at']
    search_fields = ['filename']
    readonly_fields = ['created_at']
