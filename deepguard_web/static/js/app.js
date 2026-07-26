/**
 * DeepGuard Web — Frontend JavaScript
 * Drag-and-drop upload, loading animations, frame view toggling, and metric animations.
 */

document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();
    initThresholdSlider();
    initFormSubmit();
    initFrameToggles();
    animateMetricBars();
});


/* === Drag-and-Drop Upload Zone === */
function initUploadZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-input');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeBtn = document.getElementById('remove-file');
    const analyzeBtn = document.getElementById('analyze-btn');

    if (!dropZone || !fileInput) return;

    // Click to browse
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('.remove-file')) return;
        fileInput.click();
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showFilePreview(files[0]);
        }
    });

    // File selected via browse
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            showFilePreview(fileInput.files[0]);
        }
    });

    // Remove file
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.value = '';
            filePreview.classList.remove('visible');
            analyzeBtn.disabled = true;
        });
    }

    function showFilePreview(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        const validExts = ['mp4', 'avi', 'mov', 'mkv'];

        if (!validExts.includes(ext)) {
            alert('Unsupported file format. Please use .mp4, .avi, .mov, or .mkv');
            fileInput.value = '';
            return;
        }

        if (file.size > 100 * 1024 * 1024) {
            alert('File too large. Maximum size is 100 MB.');
            fileInput.value = '';
            return;
        }

        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        filePreview.classList.add('visible');
        analyzeBtn.disabled = false;
    }
}


/* === Threshold Slider === */
function initThresholdSlider() {
    const slider = document.getElementById('threshold-slider');
    const display = document.getElementById('threshold-display');
    const hidden = document.getElementById('threshold-hidden');

    if (!slider) return;

    slider.addEventListener('input', () => {
        const val = parseFloat(slider.value).toFixed(2);
        display.textContent = val;
        hidden.value = val;
    });
}


/* === Form Submission with Loading Overlay === */
function initFormSubmit() {
    const form = document.getElementById('upload-form');
    const overlay = document.getElementById('loading-overlay');
    const analyzeBtn = document.getElementById('analyze-btn');

    if (!form || !overlay) return;

    form.addEventListener('submit', (e) => {
        const fileInput = document.getElementById('video-input');
        if (!fileInput || fileInput.files.length === 0) {
            e.preventDefault();
            alert('Please select a video file first.');
            return;
        }

        // Show loading overlay
        overlay.classList.add('active');
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;

        // Animate loading steps
        animateLoadingSteps();
    });
}


/* === Loading Step Animation === */
function animateLoadingSteps() {
    const steps = [
        { el: document.getElementById('step-1'), delay: 0 },
        { el: document.getElementById('step-2'), delay: 3000 },
        { el: document.getElementById('step-3'), delay: 6000 },
        { el: document.getElementById('step-4'), delay: 9000 },
    ];

    steps.forEach(({ el, delay }, i) => {
        if (!el) return;
        setTimeout(() => {
            el.classList.add('active');
            // Mark previous as done
            if (i > 0 && steps[i - 1].el) {
                steps[i - 1].el.classList.remove('active');
                steps[i - 1].el.classList.add('done');
            }
        }, delay);
    });
}


/* === Frame View Toggling (Crops / Heatmaps / Overlays) === */
function initFrameToggles() {
    const toggles = document.querySelectorAll('.frame-toggle');
    if (toggles.length === 0) return;

    toggles.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;

            // Update active toggle
            toggles.forEach(t => t.classList.remove('active'));
            btn.classList.add('active');

            // Show/hide frame views
            document.querySelectorAll('.frame-view').forEach(v => {
                v.style.display = 'none';
            });
            const target = document.getElementById(`view-${view}`);
            if (target) {
                target.style.display = 'grid';
            }
        });
    });
}


/* === Metric Bar Animation === */
function animateMetricBars() {
    const bars = document.querySelectorAll('.metric-bar-fill');
    if (bars.length === 0) return;

    // Use IntersectionObserver for scroll-triggered animation
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // The width is already set inline by Django template
                entry.target.style.transition = 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    bars.forEach(bar => observer.observe(bar));
}


/* === Utility Functions === */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
