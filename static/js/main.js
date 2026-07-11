// Global variables
let singleFile = null;
let compareFile = null;
let batchFiles = [];

// Tab switching functionality
document.querySelectorAll('.tab-btn').forEach(button => {
  button.addEventListener('click', () => {
    const tabName = button.getAttribute('data-tab');

    // Update active states
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');

    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');
  });
});

// ==================== SINGLE IMAGE ====================
const singleUploadArea = document.getElementById('single-upload-area');
const singleFileInput = document.getElementById('single-file-input');
const singlePreview = document.getElementById('single-preview');
const singlePlaceholder = document.getElementById('single-placeholder');
const singlePreviewContainer = document.getElementById('single-preview-container');
const singleRemoveBtn = document.getElementById('single-remove-btn');

singleUploadArea.addEventListener('click', (e) => {
  if (e.target !== singleRemoveBtn) {
    singleFileInput.click();
  }
});

singleUploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  singleUploadArea.classList.add('dragover');
});

singleUploadArea.addEventListener('dragleave', () => {
  singleUploadArea.classList.remove('dragover');
});

singleUploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  singleUploadArea.classList.remove('dragover');

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleSingleFile(files[0]);
  }
});

singleFileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleSingleFile(e.target.files[0]);
  }
});

singleRemoveBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  removeSingleFile();
});

function handleSingleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('single-error', 'Please upload an image file');
    return;
  }

  singleFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    singlePreview.src = e.target.result;
    singlePlaceholder.style.display = 'none';
    singlePreviewContainer.style.display = 'block';
  };
  reader.readAsDataURL(file);

  document.getElementById('process-btn').disabled = false;
  hideError('single-error');
}

function removeSingleFile() {
  singleFile = null;
  singleFileInput.value = '';
  singlePreview.src = '';
  singlePlaceholder.style.display = 'block';
  singlePreviewContainer.style.display = 'none';
  document.getElementById('process-btn').disabled = true;
  document.getElementById('single-results').style.display = 'none';
}

// Enhancement Sliders
const colorEnhance = document.getElementById('color-enhance');
const colorValue = document.getElementById('color-value');
const sharpnessEnhance = document.getElementById('sharpness-enhance');
const sharpnessValue = document.getElementById('sharpness-value');

colorEnhance.addEventListener('input', () => {
  colorValue.textContent = colorEnhance.value + 'x';
});

sharpnessEnhance.addEventListener('input', () => {
  sharpnessValue.textContent = sharpnessEnhance.value + 'x';
});

// Process Single Image
document.getElementById('process-btn').addEventListener('click', async () => {
  if (!singleFile) return;

  const loading = document.getElementById('single-loading');
  const results = document.getElementById('single-results');
  const errorDiv = document.getElementById('single-error');

  loading.style.display = 'block';
  results.style.display = 'none';
  hideError('single-error');

  const formData = new FormData();
  formData.append('file', singleFile);
  formData.append('color_enhance', colorEnhance.value);
  formData.append('sharpness_enhance', sharpnessEnhance.value);

  try {
    const response = await fetch('/super-resolve', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.success) {
      // Show original image
      document.getElementById('original-image').src = 'data:image/png;base64,' + data.original_image;
      document.getElementById('original-size').textContent = 'Size: ' + data.original_size;

      // Show result image
      document.getElementById('result-image').src = 'data:image/png;base64,' + data.result_image;
      document.getElementById('result-size').textContent = 'Size: ' + data.output_size;

      // Show processing info
      document.getElementById('processing-info').innerHTML = `
                Processing Time: ${data.processing_time}s | 
                Device: ${data.device} | 
                Model: ${data.model_loaded ? 'Active' : 'Not Loaded'}
            `;

      // Show download button
      document.getElementById('download-btn').style.display = 'block';
      document.getElementById('download-btn').onclick = () => downloadImage(data.result_image, 'srgan_result.png');

      results.style.display = 'block';
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      showError('single-error', 'Error: ' + data.error);
    }
  } catch (error) {
    console.error('Error:', error);
    showError('single-error', 'Failed to process image. Please try again.');
  } finally {
    loading.style.display = 'none';
  }
});

// ==================== COMPARE ====================
const compareUploadArea = document.getElementById('compare-upload-area');
const compareFileInput = document.getElementById('compare-file-input');
const comparePreview = document.getElementById('compare-preview');
const comparePlaceholder = document.getElementById('compare-placeholder');
const comparePreviewContainer = document.getElementById('compare-preview-container');
const compareRemoveBtn = document.getElementById('compare-remove-btn');

compareUploadArea.addEventListener('click', (e) => {
  if (e.target !== compareRemoveBtn) {
    compareFileInput.click();
  }
});

compareUploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  compareUploadArea.classList.add('dragover');
});

compareUploadArea.addEventListener('dragleave', () => {
  compareUploadArea.classList.remove('dragover');
});

compareUploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  compareUploadArea.classList.remove('dragover');

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleCompareFile(files[0]);
  }
});

compareFileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleCompareFile(e.target.files[0]);
  }
});

compareRemoveBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  removeCompareFile();
});

function handleCompareFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file');
    return;
  }

  compareFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    comparePreview.src = e.target.result;
    comparePlaceholder.style.display = 'none';
    comparePreviewContainer.style.display = 'block';
  };
  reader.readAsDataURL(file);

  document.getElementById('compare-btn').disabled = false;
}

function removeCompareFile() {
  compareFile = null;
  compareFileInput.value = '';
  comparePreview.src = '';
  comparePlaceholder.style.display = 'block';
  comparePreviewContainer.style.display = 'none';
  document.getElementById('compare-btn').disabled = true;
  document.getElementById('compare-results').style.display = 'none';
}

// Compare Methods
document.getElementById('compare-btn').addEventListener('click', async () => {
  if (!compareFile) return;

  const loading = document.getElementById('compare-loading');
  const results = document.getElementById('compare-results');

  loading.style.display = 'block';
  results.style.display = 'none';

  const formData = new FormData();
  formData.append('file', compareFile);

  try {
    const response = await fetch('/compare', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.success) {
      // Show all three images
      document.getElementById('compare-original').src = 'data:image/png;base64,' + data.original_image;
      document.getElementById('compare-bicubic').src = 'data:image/png;base64,' + data.bicubic_image;
      document.getElementById('compare-srgan').src = 'data:image/png;base64,' + data.srgan_image;

      // Show comparison info
      document.getElementById('compare-info').innerHTML = `
                Processing Time: ${data.processing_time}s | 
                Original: ${data.original_size} | 
                Output: ${data.output_size} | 
                Device: ${data.device}
            `;

      results.style.display = 'block';
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      alert('Error: ' + data.error);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Failed to compare images');
  } finally {
    loading.style.display = 'none';
  }
});

// ==================== BATCH ====================
const batchUploadArea = document.getElementById('batch-upload-area');
const batchFileInput = document.getElementById('batch-file-input');
const batchPreviewGrid = document.getElementById('batch-preview-grid');

batchUploadArea.addEventListener('click', () => batchFileInput.click());

batchUploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  batchUploadArea.classList.add('dragover');
});

batchUploadArea.addEventListener('dragleave', () => {
  batchUploadArea.classList.remove('dragover');
});

batchUploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  batchUploadArea.classList.remove('dragover');
  handleBatchFiles(e.dataTransfer.files);
});

batchFileInput.addEventListener('change', (e) => {
  handleBatchFiles(e.target.files);
});

function handleBatchFiles(files) {
  batchFiles = Array.from(files).filter(file => file.type.startsWith('image/'));

  if (batchFiles.length > 10) {
    alert('Maximum 10 images allowed');
    batchFiles = batchFiles.slice(0, 10);
  }

  updateBatchPreview();
  document.getElementById('batch-process-btn').disabled = batchFiles.length === 0;
}

function updateBatchPreview() {
  batchPreviewGrid.innerHTML = '';

  batchFiles.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const gridItem = document.createElement('div');
      gridItem.className = 'preview-grid-item';
      gridItem.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <div class="file-name">${file.name}</div>
                <button class="remove-grid-btn" onclick="removeBatchFile(${index})">X</button>
            `;
      batchPreviewGrid.appendChild(gridItem);
    };
    reader.readAsDataURL(file);
  });
}

function removeBatchFile(index) {
  batchFiles.splice(index, 1);
  updateBatchPreview();
  document.getElementById('batch-process-btn').disabled = batchFiles.length === 0;

  if (batchFiles.length === 0) {
    document.getElementById('batch-results').style.display = 'none';
  }
}

// Batch Process
document.getElementById('batch-process-btn').addEventListener('click', async () => {
  if (batchFiles.length === 0) return;

  const loading = document.getElementById('batch-loading');
  const results = document.getElementById('batch-results');
  const gallery = document.getElementById('batch-gallery');

  loading.style.display = 'block';
  results.style.display = 'none';
  gallery.innerHTML = '';

  for (let i = 0; i < batchFiles.length; i++) {
    const file = batchFiles[i];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('color_enhance', document.getElementById('batch-color').value);
    formData.append('sharpness_enhance', document.getElementById('batch-sharpness').value);

    try {
      const response = await fetch('/super-resolve', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        // Show original
        const originalReader = new FileReader();
        originalReader.onload = (e) => {
          const galleryItem = document.createElement('div');
          galleryItem.className = 'gallery-item';
          galleryItem.innerHTML = `
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px;">
                            <div>
                                <small style="color: #666;">Original</small>
                                <img src="${e.target.result}" alt="Original" style="width: 100%; height: 150px; object-fit: cover; border-radius: 5px;">
                            </div>
                            <div>
                                <small style="color: #666;">SRGAN Result</small>
                                <img src="data:image/png;base64,${data.result_image}" alt="Result" style="width: 100%; height: 150px; object-fit: cover; border-radius: 5px;">
                            </div>
                        </div>
                        <div class="gallery-info" style="padding: 10px; font-size: 12px; color: #666;">
                            ${file.name} | Time: ${data.processing_time}s | ${data.output_size}
                        </div>
                    `;
          gallery.appendChild(galleryItem);
        };
        originalReader.readAsDataURL(file);
      }
    } catch (error) {
      console.error('Error processing file:', file.name, error);
    }
  }

  if (gallery.children.length > 0) {
    document.getElementById('batch-download-btn').style.display = 'block';
  }

  results.style.display = 'block';
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  loading.style.display = 'none';
});

// ==================== UTILITY FUNCTIONS ====================
function downloadImage(base64Data, filename) {
  const link = document.createElement('a');
  link.href = 'data:image/png;base64,' + base64Data;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function showError(elementId, message) {
  const errorDiv = document.getElementById(elementId);
  errorDiv.textContent = message;
  errorDiv.style.display = 'block';
}

function hideError(elementId) {
  const errorDiv = document.getElementById(elementId);
  errorDiv.style.display = 'none';
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize on load
window.addEventListener('load', () => {
  console.log('SRGAN Web Application loaded');
  console.log('Device:', document.getElementById('device-info').textContent);

  // Check model status
  fetch('/model-status')
    .then(response => response.json())
    .then(data => {
      console.log('Model status:', data);
      if (!data.model_loaded) {
        console.warn('Model not loaded. Please ensure SRGAN Generator Model.pth exists');
      }
    })
    .catch(error => console.error('Error checking model status:', error));
});