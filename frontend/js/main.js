// Backend API URL - Hugging Face Space deployment
const BACKEND_URL = 'https://moeenuddin01-plantdoc-ai.hf.space/v1/prediction/';

// DOM Elements
const uploadCard = document.getElementById('uploadCard');
const fileInput = document.getElementById('fileInput');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeImageBtn = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const diseaseName = document.getElementById('diseaseName');
const confidenceValue = document.getElementById('confidenceValue');
const progressCircle = document.getElementById('progressCircle');
const treatmentContent = document.getElementById('treatmentContent');

// Selected file
let selectedFile = null;

// Drag and drop event handlers
uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.classList.add('drag-over');
});

uploadCard.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadCard.classList.remove('drag-over');
});

uploadCard.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadCard.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Click to upload
uploadCard.addEventListener('click', (e) => {
    if (e.target === uploadCard || e.target === uploadContent) {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file selection
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a JPG or PNG image.');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB.');
        return;
    }

    selectedFile = file;
    hideError();
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Remove image
removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearUpload();
});

function clearUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadContent.style.display = 'block';
    previewContainer.style.display = 'none';
    hideError();
    hideResults();
}

// Analyze button click
analyzeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    analyzeImage();
});

// Analyze image
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading state
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;
    hideError();

    // Convert image to base64
    const base64Image = await fileToBase64(selectedFile);

    try {
        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_base64: base64Image
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Analysis error:', error);
        showError('Failed to analyze image. Please make sure the backend server is running and try again.');
    } finally {
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.disabled = false;
    }
}

// Convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            // Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
            const base64String = reader.result.split(',')[1];
            resolve(base64String);
        };
        reader.onerror = (error) => {
            reject(error);
        };
    });
}

// Display results
function displayResults(data) {
    // Set disease name
    diseaseName.textContent = data.predicted_class || data.disease_name || data.disease || 'Unknown Disease';
    
    // Animate confidence ring (convert decimal to percentage)
    const confidence = data.confidence || data.confidence_percent || 0;
    const confidencePercent = confidence <= 1 ? confidence * 100 : confidence;
    animateConfidence(confidencePercent);
    
    // Set treatment tips from RAG + Groq LLM response
    if (data.treatment_recommendations) {
        // Convert markdown-like formatting to HTML
        const formattedTreatment = data.treatment_recommendations
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')  // Italic
            .replace(/\n\n/g, '</p><p>')  // Paragraphs
            .replace(/\n/g, '<br>')  // Line breaks
            .replace(/^/, '<p>')  // Open first paragraph
            .replace(/$/, '</p>');  // Close last paragraph
        treatmentContent.innerHTML = formattedTreatment;
    } else {
        treatmentContent.innerHTML = '<p><em>AI-powered treatment recommendations are currently unavailable. Please consult an agricultural extension service for treatment guidance.</em></p>';
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Animate confidence ring
function animateConfidence(percentage) {
    const radius = 65;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percentage / 100) * circumference;
    
    // Reset to 0
    progressCircle.style.strokeDashoffset = circumference;
    
    // Animate to target
    setTimeout(() => {
        progressCircle.style.strokeDashoffset = offset;
    }, 100);
    
    // Animate number counter
    animateNumber(confidenceValue, 0, Math.round(percentage), 1000);
    
    // Remove previous confidence classes
    progressCircle.classList.remove('confidence-high', 'confidence-medium', 'confidence-low');
    
    // Apply appropriate confidence class and label
    const confidenceLabel = document.getElementById('confidenceLabel');
    confidenceLabel.classList.remove('high', 'medium', 'low');
    
    if (percentage >= 75) {
        progressCircle.classList.add('confidence-high');
        confidenceLabel.textContent = 'High Confidence';
        confidenceLabel.classList.add('high');
    } else if (percentage >= 50) {
        progressCircle.classList.add('confidence-medium');
        confidenceLabel.textContent = 'Medium Confidence';
        confidenceLabel.classList.add('medium');
    } else {
        progressCircle.classList.add('confidence-low');
        confidenceLabel.textContent = 'Low Confidence';
        confidenceLabel.classList.add('low');
    }
}

// Animate number counter
function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = Math.round(start + (end - start) * easeOutQuart);
        
        element.textContent = current;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Hide results section
function hideResults() {
    resultsSection.style.display = 'none';
}

// Prevent default drag behavior on document
document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});
