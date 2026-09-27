const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const predictButton = document.getElementById('predictButton');
const clearButton = document.getElementById('clearButton');
const statusBox = document.getElementById('statusBox');
const resultCard = document.getElementById('resultCard');
const resultDigit = document.getElementById('resultDigit');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');

let selectedFile = null;

const setStatus = (message, isError = false) => {
  statusBox.textContent = message;
  statusBox.style.color = isError ? '#ff9a9a' : '#b7c7de';
  statusBox.style.borderColor = isError ? 'rgba(255, 107, 107, 0.5)' : 'rgba(148, 163, 184, 0.22)';
};

const resetResult = () => {
  resultCard.classList.add('hidden');
  resultDigit.textContent = '-';
  confidenceFill.style.width = '0%';
  confidenceText.textContent = '0%';
};

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  selectedFile = file || null;

  if (!selectedFile) {
    previewImage.hidden = true;
    previewImage.removeAttribute('src');
    resetResult();
    setStatus('No image uploaded yet.');
    return;
  }

  const objectUrl = URL.createObjectURL(selectedFile);
  previewImage.src = objectUrl;
  previewImage.hidden = false;
  resetResult();
  setStatus(`Selected: ${selectedFile.name}`);
});

predictButton.addEventListener('click', async () => {
  if (!selectedFile) {
    setStatus('Please choose an image first.', true);
    return;
  }

  const formData = new FormData();
  formData.append('image', selectedFile);

  predictButton.disabled = true;
  setStatus('Analyzing image...');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Unable to predict the digit.');
    }

    const confidencePercentage = Math.round((data.confidence || 0) * 100);
    resultDigit.textContent = data.digit;
    confidenceFill.style.width = `${confidencePercentage}%`;
    confidenceText.textContent = `${confidencePercentage}%`;
    resultCard.classList.remove('hidden');
    setStatus(`Prediction complete with ${confidencePercentage}% confidence.`);
  } catch (error) {
    setStatus(error.message || 'Prediction failed.', true);
    resetResult();
  } finally {
    predictButton.disabled = false;
  }
});

clearButton.addEventListener('click', () => {
  imageInput.value = '';
  selectedFile = null;
  previewImage.hidden = true;
  previewImage.removeAttribute('src');
  resetResult();
  setStatus('No image uploaded yet.');
});
