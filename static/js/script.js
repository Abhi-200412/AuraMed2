// static/js/script.js - Premium UI/UX Interactions

document.addEventListener('DOMContentLoaded', () => {

    // ====================== ELEMENTS ======================
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const fileInput = document.getElementById('fileInput');

    // ====================== LOADING OVERLAY ======================
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-spinner';
    loadingOverlay.innerHTML = `
        <div class="text-center fade-in-up">
            <div class="custom-loader mx-auto"></div>
            <h3 class="fw-bold mb-2 text-white">Analyzing Image...</h3>
            <p class="mb-0" style="color: var(--primary);">Running Deep Learning Model • Please wait</p>
            
            <div class="mt-4">
                <div class="progress mx-auto" style="height: 6px; max-width: 250px; background: rgba(255,255,255,0.1);">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         style="width: 100%; background: var(--primary);"></div>
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

            // Show beautiful loading overlay
            loadingOverlay.classList.add('show');

            // Disable button
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalHTML = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-3" role="status"></span>
                    Processing...
                `;

                // Safety timeout (in case of network failure)
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

    // ====================== FILE INPUT DRAG & DROP ======================
    if (fileInput) {
        const fileContainer = fileInput.parentElement;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileInput.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            fileInput.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileInput.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            fileInput.style.borderColor = 'var(--primary)';
            fileInput.style.background = 'rgba(0, 242, 254, 0.1)';
        }

        function unhighlight(e) {
            fileInput.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            fileInput.style.background = 'rgba(255, 255, 255, 0.03)';
        }

        const maxFiles = 10;
        let dt = new DataTransfer();

        function renderPreviews() {
            const previewContainer = document.getElementById('imagePreviewGrid');
            if (!previewContainer) return;
            previewContainer.innerHTML = '';

            Array.from(fileInput.files).forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = function (e) {
                    const col = document.createElement('div');
                    col.className = 'col-auto position-relative fade-in-up';
                    col.style.animationDelay = `${index * 0.05}s`;

                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'img-thumbnail border-secondary';
                    img.style.height = '100px';
                    img.style.width = '100px';
                    img.style.objectFit = 'cover';
                    img.style.borderRadius = '12px';
                    img.style.background = 'rgba(255,255,255,0.05)';
                    img.title = file.name;

                    const removeBtn = document.createElement('button');
                    removeBtn.innerHTML = '<i class="fas fa-times"></i>';
                    removeBtn.className = 'btn btn-sm btn-danger rounded-circle position-absolute';
                    removeBtn.style.top = '-5px';
                    removeBtn.style.right = '-5px';
                    removeBtn.style.width = '24px';
                    removeBtn.style.height = '24px';
                    removeBtn.style.padding = '0';
                    removeBtn.style.display = 'flex';
                    removeBtn.style.alignItems = 'center';
                    removeBtn.style.justifyContent = 'center';
                    removeBtn.style.boxShadow = '0 2px 5px rgba(0,0,0,0.5)';
                    removeBtn.type = 'button';

                    removeBtn.onclick = function (ev) {
                        ev.preventDefault();
                        ev.stopPropagation();
                        // Rebuild FileList without this file
                        const newDt = new DataTransfer();
                        Array.from(fileInput.files).forEach((f, i) => {
                            if (i !== index) newDt.items.add(f);
                        });
                        fileInput.files = newDt.files;
                        dt = newDt;
                        renderPreviews();
                    };

                    col.appendChild(img);
                    col.appendChild(removeBtn);
                    previewContainer.appendChild(col);
                };
                reader.readAsDataURL(file);
            });
        }

        fileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                if (this.files.length > maxFiles) {
                    flashMessage(`Too many files! You selected ${this.files.length} images but the maximum allowed is ${maxFiles}.`, 'warning');
                    this.value = '';
                    dt = new DataTransfer();
                    renderPreviews();
                    return;
                }

                dt = new DataTransfer();
                Array.from(this.files).forEach(f => dt.items.add(f));
                fileInput.files = dt.files;

                flashMessage(`Great! ${this.files.length} file(s) loaded up for analysis.`, 'info');
                renderPreviews();
            } else {
                dt = new DataTransfer();
                renderPreviews();
            }
        });
    }

    // ====================== FLASH MESSAGES ======================
    function flashMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show shadow-lg position-fixed top-0 start-50 translate-middle-x mt-4`;
        alertDiv.style.zIndex = '1050';
        alertDiv.style.minWidth = '320px';
        alertDiv.innerHTML = `
            <i class="fas ${type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle'} me-2"></i> ${message}
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);

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

    // ====================== KEYBOARD SHORTCUTS ======================
    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
            e.preventDefault();
            fileInput?.focus();
        }

        if (e.key === "Escape" && loadingOverlay.classList.contains('show')) {
            loadingOverlay.classList.remove('show');
        }
    });

    window.addEventListener('beforeunload', () => {
        if (loadingOverlay.classList.contains('show')) {
            loadingOverlay.classList.remove('show');
        }
    });

    // Subtly animate result cards sequentially
    const cards = document.querySelectorAll('.result-card');
    if (cards.length > 0) {
        cards.forEach((card, index) => {
            card.style.animationDelay = `${(index + 3) * 0.15}s`;
        });
    }
});