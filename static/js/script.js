// static/js/script.js - Full Updated Version (Modern + Smooth)

document.addEventListener('DOMContentLoaded', () => {

    // ====================== ELEMENTS ======================
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const fileInput = document.getElementById('fileInput');

    // ====================== LOADING OVERLAY ======================
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-spinner';
    loadingOverlay.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-info mb-4" 
                 style="width: 5.5rem; height: 5.5rem; border-width: 8px;" 
                 role="status"></div>
            <h3 class="text-light fw-bold mb-2">Analyzing Image...</h3>
            <p class="text-info mb-0">Running SBCAE Model • Please wait</p>
            
            <div class="mt-4">
                <div class="progress" style="height: 6px; max-width: 280px; margin: 0 auto;">
                    <div class="progress-bar bg-info progress-bar-striped progress-bar-animated" 
                         style="width: 100%;"></div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(loadingOverlay);

    // ====================== FORM SUBMISSION ======================
    if (uploadForm) {
        uploadForm.addEventListener('submit', function (e) {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                flashMessage('Please select an image to upload!', 'warning');
                return;
            }

            // Show beautiful loading
            loadingOverlay.classList.add('show');

            // Disable button with animation
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalHTML = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-3" role="status"></span>
                    Processing Image...
                `;

                // Safety timeout (in case something goes wrong)
                setTimeout(() => {
                    if (loadingOverlay.classList.contains('show')) {
                        loadingOverlay.classList.remove('show');
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = originalHTML;
                    }
                }, 30000);
            }
        });
    }

    // ====================== FILE INPUT PREVIEW ======================
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (this.files && this.files[0]) {
                const fileName = this.files[0].name;
                if (fileName.length > 40) {
                    this.title = fileName;
                } else {
                    this.title = '';
                }
            }
        });
    }

    // ====================== FLASH MESSAGES ======================
    function flashMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show shadow-sm position-fixed top-0 start-50 translate-middle-x mt-4`;
        alertDiv.style.zIndex = '9999';
        alertDiv.style.minWidth = '320px';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);

        // Auto dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                const bsAlert = new bootstrap.Alert(alertDiv);
                bsAlert.close();
            }
        }, 5000);
    }

    // ====================== AUTO DISMISS EXISTING ALERTS ======================
    setTimeout(() => {
        document.querySelectorAll('.alert').forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 6500);

    // ====================== KEYBOARD SHORTCUT ======================
    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
            e.preventDefault();
            fileInput?.focus();
        }

        // Press ESC to hide loading (emergency)
        if (e.key === "Escape" && loadingOverlay.classList.contains('show')) {
            loadingOverlay.classList.remove('show');
        }
    });

    // ====================== CLEANUP ON PAGE UNLOAD ======================
    window.addEventListener('beforeunload', () => {
        if (loadingOverlay.classList.contains('show')) {
            loadingOverlay.classList.remove('show');
        }
    });

    console.log('✅ Enhanced JS loaded successfully');
});