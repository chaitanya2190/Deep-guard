from django.db import models


class AnalysisResult(models.Model):
    """Stores the results of each deepfake detection analysis."""

    VERDICT_CHOICES = [
        ('AUTHENTIC', 'Authentic'),
        ('MANIPULATED', 'Manipulated'),
    ]

    video_file = models.FileField(upload_to='uploads/%Y/%m/')
    filename = models.CharField(max_length=255)
    deepfake_probability = models.FloatField(help_text='Percentage 0-100')
    reconstruction_error = models.FloatField()
    is_anomaly = models.BooleanField(default=False)
    anomaly_threshold = models.FloatField(default=0.15)
    verdict = models.CharField(max_length=20, choices=VERDICT_CHOICES)
    explanation = models.TextField(blank=True)
    frame_crops_json = models.TextField(
        blank=True,
        help_text='JSON array of base64-encoded face crop thumbnails'
    )
    heatmaps_json = models.TextField(
        blank=True,
        help_text='JSON array of base64-encoded heatmap images'
    )
    processing_time = models.FloatField(null=True, help_text='Seconds')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.filename} — {self.verdict} ({self.deepfake_probability}%)"

    @property
    def is_fake(self):
        return self.verdict == 'MANIPULATED'

    @property
    def probability_display(self):
        return f"{self.deepfake_probability}%"
