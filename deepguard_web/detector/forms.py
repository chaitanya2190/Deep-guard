from django import forms


class VideoUploadForm(forms.Form):
    """Form for video file upload with anomaly threshold configuration."""
    video = forms.FileField(
        label='Upload Video',
        help_text='Supported formats: .mp4, .avi, .mov, .mkv',
    )
    threshold = forms.FloatField(
        label='Anomaly Threshold',
        initial=0.15,
        min_value=0.01,
        max_value=0.50,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'id': 'threshold-input',
        }),
    )

    def clean_video(self):
        video = self.cleaned_data.get('video')
        if video:
            ext = video.name.rsplit('.', 1)[-1].lower()
            if ext not in ('mp4', 'avi', 'mov', 'mkv'):
                raise forms.ValidationError('Unsupported video format. Use .mp4, .avi, .mov, or .mkv')
            if video.size > 100 * 1024 * 1024:  # 100 MB
                raise forms.ValidationError('File too large. Maximum size is 100 MB.')
        return video
