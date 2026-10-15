const $ = (id) => document.getElementById(id);
const input = $('file-input');
const dropZone = $('drop-zone');
const analyze = $('analyze');
let selectedFile = null;
let previewUrl = null;
let latestResult = null;
let busy = false;

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}

function clearResult() {
  latestResult = null;
  $('result').hidden = true;
  $('loading-state').hidden = true;
  $('empty-state').hidden = false;
  $('result-status').textContent = 'Awaiting image';
  $('result-status').classList.remove('complete');
}

function selectFile(file) {
  if (busy || !file) return;
  showError('');
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    showError('Please choose a JPEG or PNG image.');
    input.value = '';
    return;
  }
  if (file.size === 0 || file.size > 10 * 1024 * 1024) {
    showError(file.size === 0 ? 'This file is empty. Choose another image.' : 'Please choose an image smaller than 10 MB.');
    input.value = '';
    return;
  }
  clearResult();
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('preview').src = previewUrl;
  $('preview').hidden = false;
  $('preview-label').hidden = false;
  $('upload-prompt').hidden = true;
  $('file-details').hidden = false;
  $('file-name').textContent = file.name;
  $('file-size').textContent = `${(file.size / 1024).toFixed(1)} KB`;
  analyze.disabled = false;
  $('result-status').textContent = 'Ready to analyze';
}

dropZone.addEventListener('click', () => { if (!busy) input.click(); });
input.addEventListener('change', () => selectFile(input.files[0]));
['dragenter', 'dragover'].forEach((event) => dropZone.addEventListener(event, (e) => {
  e.preventDefault();
  if (!busy) dropZone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach((event) => dropZone.addEventListener(event, (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
}));
dropZone.addEventListener('drop', (e) => selectFile(e.dataTransfer.files[0]));
$('remove-file').addEventListener('click', () => {
  if (busy) return;
  selectedFile = null;
  input.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('preview').removeAttribute('src');
  $('preview').hidden = true;
  $('preview-label').hidden = true;
  $('upload-prompt').hidden = false;
  $('file-details').hidden = true;
  analyze.disabled = true;
  clearResult();
  showError('');
});

analyze.addEventListener('click', async () => {
  if (!selectedFile || busy) return;
  busy = true;
  clearResult();
  showError('');
  analyze.disabled = true;
  dropZone.disabled = true;
  $('remove-file').disabled = true;
  analyze.firstElementChild.textContent = 'Analyzing…';
  $('empty-state').hidden = true;
  $('loading-state').hidden = false;
  $('result-status').textContent = 'Processing';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const form = new FormData();
    form.append('file', selectedFile);
    const response = await fetch('/predict', { method: 'POST', body: form, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : 'Analysis failed. Please try again.');
    if (!data || !['NORMAL', 'PNEUMONIA'].includes(data.prediction) || typeof data.confidence !== 'number' || !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1) {
      throw new Error('The server returned an unexpected result. Please try again.');
    }
    latestResult = { file: selectedFile.name, analyzed_at: new Date().toISOString(), ...data };
    const pneumonia = data.prediction === 'PNEUMONIA';
    $('prediction').textContent = pneumonia ? 'Pneumonia' : 'Normal';
    $('result').classList.toggle('pneumonia', pneumonia);
    $('result-description').textContent = pneumonia ? 'The model classified this image as pneumonia.' : 'The model classified this image as normal.';
    $('pneumonia-value').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    $('normal-value').textContent = `${((1 - data.confidence) * 100).toFixed(1)}%`;
    $('pneumonia-bar').value = data.confidence;
    $('normal-bar').value = 1 - data.confidence;
    $('result').hidden = false;
    $('result-status').textContent = 'Analysis complete';
    $('result-status').classList.add('complete');
  } catch (error) {
    $('empty-state').hidden = false;
    $('result-status').textContent = 'Analysis failed';
    showError(error.name === 'AbortError' ? 'The request timed out. Please try again.' : error instanceof TypeError ? 'Unable to reach the server. Check your connection and try again.' : error.message);
  } finally {
    clearTimeout(timeout);
    busy = false;
    $('loading-state').hidden = true;
    analyze.disabled = !selectedFile;
    dropZone.disabled = false;
    $('remove-file').disabled = false;
    analyze.firstElementChild.textContent = 'Analyze X-ray';
  }
});

$('download').addEventListener('click', () => {
  if (!latestResult) return;
  const blob = new Blob([JSON.stringify(latestResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'pulmo-analysis.json';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
