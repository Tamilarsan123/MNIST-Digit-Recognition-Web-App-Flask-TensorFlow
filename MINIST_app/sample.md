You are an expert Python Full-Stack Developer, Machine Learning Engineer, UI/UX Designer, and Production Deployment Engineer.

Build a complete, production-quality **MNIST Handwritten Digit Recognition Web Application** using **Python + Flask + TensorFlow/Keras + HTML/CSS/JavaScript**.

The application must allow a user to upload an image containing a handwritten digit, process the image exactly according to the trained MNIST model's preprocessing pipeline, predict the digit from 0–9, display the prediction with confidence, and provide a clear/reset option to remove the previous prediction.

IMPORTANT:
Do not replace the trained model with another architecture.
Do not retrain the model inside the web application.
Use the provided trained model file:

`digits_intel.h5`

The trained model was created using the following architecture:

```python
model = Sequential()
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

The model was compiled with:

```python
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
```

The MNIST training images are:

```text
28 x 28
grayscale
pixel values: 0–255
```

The original preprocessing used during prediction is:

```python
image = image.convert(mode='L')
image = image.resize(size=(28, 28))
image = np.array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)
```

Prediction is obtained using:

```python
prediction = model.predict(image)
digit = np.argmax(prediction)
```

Therefore, the Flask application must preserve this preprocessing pipeline as closely as possible.

==================================================

1. PROJECT GOAL
   ==================================================

Create a beautiful, modern, responsive web application called:

**DigitAI — MNIST Handwritten Digit Recognition**

The application should look like a premium AI/ML product rather than a basic student project.

Main user flow:

1. User opens the website.
2. User sees an attractive landing/prediction interface.
3. User uploads an image of a handwritten digit.
4. Frontend shows a preview of the uploaded image.
5. User clicks "Predict Digit".
6. Flask receives the image.
7. Backend validates the uploaded file.
8. Backend converts image to grayscale.
9. Backend resizes it to 28×28.
10. Backend converts it to a NumPy array.
11. Backend normalizes pixel values by dividing by 255.
12. Backend adds the batch dimension.
13. Model predicts probabilities for digits 0–9.
14. Backend calculates the predicted digit using `argmax`.
15. Backend calculates confidence from the corresponding softmax probability.
16. Frontend displays the prediction beautifully.
17. User can click "Clear" to remove the uploaded image and previous prediction.
18. User can upload another image and predict again.

==================================================
2. TECHNOLOGY STACK
===================

Backend:

* Python 3.x
* Flask
* TensorFlow / Keras
* NumPy
* Pillow
* Werkzeug
* python-dotenv if environment configuration is needed

Frontend:

* HTML5
* CSS3
* Vanilla JavaScript
* No unnecessary frontend framework
* Use modern CSS
* Responsive design
* Accessible UI

Model:

* TensorFlow/Keras
* Existing `digits_intel.h5`

Deployment:

* Flask production deployment
* Gunicorn for Linux/cloud deployment
* `requirements.txt`
* `Procfile` if useful for hosting platforms
* `.gitignore`
* environment variables where appropriate

==================================================
3. PRODUCTION-GRADE FOLDER STRUCTURE
====================================

Create this folder structure:

```text
mnist-digit-recognition/
│
├── app/
│   ├── __init__.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── main_routes.py
│   │   └── prediction_routes.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py
│   │   └── image_service.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── validators.py
│   │
│   ├── templates/
│   │   ├── base.html
│   │   └── index.html
│   │
│   └── static/
│       ├── css/
│       │   └── style.css
│       │
│       ├── js/
│       │   └── app.js
│       │
│       └── assets/
│
├── models/
│   └── digits_intel.h5
│
├── uploads/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_routes.py
│   ├── test_image_service.py
│   └── test_prediction.py
│
├── config.py
├── run.py
├── requirements.txt
├── .env.example
├── .gitignore
├── Procfile
├── README.md
└── LICENSE
```

Keep the architecture modular.

Do NOT put all Flask logic into one `app.py` file.

==================================================
4. APPLICATION ARCHITECTURE
===========================

Use this flow:

```text
User
  ↓
Browser
  ↓
HTML/CSS/JavaScript
  ↓
Flask Route
  ↓
File Validation
  ↓
Image Processing Service
  ↓
28×28 Grayscale Image
  ↓
Normalization /255.0
  ↓
Batch Dimension
  ↓
Keras Model
  ↓
Prediction Probabilities
  ↓
Argmax
  ↓
Predicted Digit + Confidence
  ↓
JSON Response
  ↓
Beautiful Result UI
```

Separate responsibilities properly.

`prediction_routes.py`

* Receive upload
* Validate request
* Call image service
* Call model service
* Return prediction response

`image_service.py`

* Open image using Pillow
* Convert to grayscale
* Resize to 28×28
* Convert to NumPy
* Normalize pixels
* Add batch dimension

`model_service.py`

* Load model once
* Never reload the model for every request
* Perform prediction
* Return predicted digit
* Return confidence
* Optionally return probabilities for all 10 digits

==================================================
5. MODEL LOADING
================

Load the model only once when the Flask application starts.

Use something conceptually similar to:

```python
from tensorflow.keras.models import load_model

model = load_model("models/digits_intel.h5")
```

Do NOT call `load_model()` inside every prediction request.

Create a reusable model service.

Example responsibility:

```text
ModelService
    ├── load_model()
    ├── predict()
    └── get_prediction_details()
```

Handle model-loading errors gracefully.

If the model file does not exist, display/log a clear configuration error instead of crashing with an unclear traceback.

==================================================
6. IMAGE PREPROCESSING
======================

This is extremely important.

The application must reproduce the preprocessing used by the notebook.

For every uploaded image:

1. Open using Pillow.
2. Convert to grayscale:

```python
image.convert("L")
```

3. Resize:

```python
image.resize((28, 28))
```

4. Convert to NumPy:

```python
np.array(image)
```

5. Normalize:

```python
image = image / 255.0
```

6. Add batch dimension:

```python
image = np.expand_dims(image, axis=0)
```

The final model input must be:

```text
(1, 28, 28)
```

Do not unnecessarily change the input shape to `(1, 28, 28, 1)` because the supplied model was trained using Flatten on `(28, 28)` input.

Preserve compatibility with the trained model.

==================================================
7. IMPORTANT MNIST IMAGE HANDLING
=================================

MNIST images normally have a black background and white handwritten digit.

User-uploaded images may be:

* black digit on white background
* white digit on black background
* colored
* grayscale
* screenshot
* PNG
* JPG
* JPEG
* WEBP

Support common image formats safely.

However, do not blindly invert every image.

Create preprocessing logic that attempts to maintain compatibility with the model.

Prefer the simplest preprocessing that matches the original notebook:

```text
grayscale
→ resize 28×28
→ NumPy array
→ /255
→ batch dimension
```

Do not introduce advanced image processing unless necessary.

==================================================
8. FILE VALIDATION
==================

Implement secure upload validation.

Allowed formats:

```text
PNG
JPG
JPEG
WEBP
```

Maximum file size:

```text
5 MB
```

Reject:

* unsupported extensions
* empty files
* files larger than the limit
* invalid/corrupted images

Never trust the original filename.

Use secure filename handling.

Do not allow path traversal.

Do not execute uploaded files.

==================================================
9. API DESIGN
=============

Create a prediction endpoint:

```text
POST /api/predict
```

Request:

```text
multipart/form-data
```

Field:

```text
image
```

Successful response:

```json
{
    "success": true,
    "digit": 7,
    "confidence": 0.9821,
    "probabilities": {
        "0": 0.0001,
        "1": 0.0002,
        "2": 0.0003,
        "3": 0.0010,
        "4": 0.0001,
        "5": 0.0020,
        "6": 0.0001,
        "7": 0.9821,
        "8": 0.0030,
        "9": 0.0111
    }
}
```

The probabilities should come directly from the model's softmax output.

Return confidence as a clean percentage on the frontend.

For errors:

```json
{
    "success": false,
    "error": "Please upload a valid image."
}
```

Use appropriate HTTP status codes.

==================================================
10. FRONTEND DESIGN
===================

Create a premium AI dashboard UI.

Application name:

**DigitAI**

Subtitle:

**MNIST Handwritten Digit Recognition**

Hero heading:

**Turn Your Handwritten Digit Into AI Prediction**

Supporting text:

**Upload an image of a handwritten digit and let our trained neural network identify it instantly.**

Use a premium modern visual style.

Color direction:

* Deep navy / near-black background
* Electric blue
* Soft violet
* Cyan accents
* White / off-white text
* Subtle gradients
* Glassmorphism cards
* Soft shadows
* Thin borders
* Premium spacing

Avoid excessive gradients.

Do not make the UI look childish.

Do not use too many colors.

The design should resemble a modern AI SaaS dashboard.

==================================================
11. PAGE LAYOUT
===============

Create one primary page.

Header:

```text
DigitAI
MNIST Digit Recognition

[GitHub]
```

Hero section:

```text
AI-POWERED
HANDWRITTEN DIGIT RECOGNITION

Turn Your Handwritten Digit
Into an AI Prediction

Upload an image and let the neural network recognize
the handwritten number in seconds.
```

Main prediction card:

```text
┌────────────────────────────────────────────┐
│                                            │
│             Upload Your Digit              │
│                                            │
│       Drag & Drop Image Here               │
│                or                          │
│          [ Browse Image ]                  │
│                                            │
│       PNG • JPG • JPEG • WEBP              │
│       Maximum size: 5 MB                   │
│                                            │
└────────────────────────────────────────────┘
```

After upload, show:

```text
Image Preview

[ uploaded image ]

filename.jpg

[ Predict Digit ]   [ Clear ]
```

==================================================
12. DRAG AND DROP
=================

Support:

* Click to browse
* Drag and drop
* File selection
* Image preview

The drop zone should visually react when the user drags an image over it.

Example:

```text
Drag image here
↓
Drop to upload
```

Add a subtle animation.

==================================================
13. PREDICTION RESULT
=====================

After prediction, show a beautiful result card.

Example:

```text
PREDICTION RESULT

             7

        Seven

Confidence

██████████████████░░ 98.21%

The model is highly confident that
the uploaded image represents digit 7.
```

Make the predicted number very large.

Example:

```text
7
```

Use animated appearance.

Do not claim that the model is "100% accurate."

Use the actual prediction confidence.

==================================================
14. PROBABILITY VISUALIZATION
=============================

Display probabilities for all digits.

Example:

```text
Digit Probabilities

0  █
1  █
2  ██
3  █
4  █
5  ██
6  █
7  ███████████████████
8  ██
9  ███
```

Use horizontal progress bars.

Show:

```text
0   0.01%
1   0.02%
...
7   98.21%
```

Sort or display digits 0–9 consistently.

Do not hide the other class probabilities.

==================================================
15. CLEAR FUNCTIONALITY
=======================

The Clear button is mandatory.

When the user clicks Clear:

* Remove image preview
* Remove filename
* Reset file input
* Remove prediction
* Remove confidence
* Remove probability bars
* Hide result card
* Return UI to upload state
* Remove any error messages
* Allow another image to be uploaded

The clear operation should happen instantly without refreshing the entire page.

==================================================
16. LOADING STATE
=================

When the user clicks Predict:

Disable the prediction button temporarily.

Show:

```text
Analyzing...
```

Add a small animated loader.

After response:

* Hide loader
* Re-enable button
* Display result

Prevent duplicate requests while prediction is running.

==================================================
17. ERROR STATES
================

Handle errors elegantly.

Examples:

No image:

```text
Please upload an image first.
```

Invalid file:

```text
Please upload a PNG, JPG, JPEG, or WEBP image.
```

Large file:

```text
Image size must be less than 5 MB.
```

Prediction error:

```text
Something went wrong while analyzing the image.
Please try again.
```

Model error:

```text
The prediction service is temporarily unavailable.
```

Do not expose Python tracebacks to users.

Log detailed errors server-side.

==================================================
18. RESPONSIVE DESIGN
=====================

The application must work properly on:

* Desktop
* Laptop
* Tablet
* Mobile

Desktop:

Two-column layout where appropriate:

```text
Upload / Preview       Prediction Result
```

Mobile:

Single-column layout.

Buttons should be touch-friendly.

Do not allow horizontal scrolling.

==================================================
19. ACCESSIBILITY
=================

Follow basic accessibility practices.

Use:

* Semantic HTML
* Labels
* Alt text
* Keyboard-accessible controls
* Visible focus states
* Sufficient contrast
* ARIA attributes where useful

File upload should be accessible by keyboard.

==================================================
20. JAVASCRIPT ARCHITECTURE
===========================

Do not write one huge JavaScript function.

Use modular functions such as:

```javascript
handleFileSelect()
handleDragOver()
handleDrop()
previewImage()
predictDigit()
displayPrediction()
displayProbabilities()
showLoading()
hideLoading()
showError()
clearPrediction()
resetApplication()
```

Use:

```javascript
fetch("/api/predict", {
    method: "POST",
    body: formData
})
```

Do not reload the page during prediction.

==================================================
21. SECURITY
============

Implement basic production security.

Requirements:

* `secure_filename`
* file extension validation
* MIME/type validation where practical
* maximum upload size
* no arbitrary file execution
* no user-controlled filesystem paths
* safe error messages
* debug mode disabled in production
* secrets through environment variables
* do not hard-code API keys
* do not expose server stack traces

Configure:

```text
MAX_CONTENT_LENGTH
```

to approximately 5 MB.

==================================================
22. FLASK CONFIGURATION
=======================

Create configuration separation.

For example:

```text
DevelopmentConfig
ProductionConfig
TestingConfig
```

Environment variables should control:

```text
FLASK_ENV
SECRET_KEY
MODEL_PATH
MAX_CONTENT_LENGTH
```

Do not commit `.env`.

Create:

```text
.env.example
```

==================================================
23. LOGGING
===========

Implement useful server-side logging.

Log:

* application startup
* model loading
* prediction request
* prediction success
* validation errors
* unexpected exceptions

Do not log sensitive user information unnecessarily.

==================================================
24. TESTING
===========

Create basic automated tests.

Test:

1. Home page loads.
2. Prediction endpoint rejects missing image.
3. Prediction endpoint rejects invalid file.
4. Prediction endpoint accepts valid image.
5. Image preprocessing returns expected shape.
6. Model service returns digit between 0 and 9.
7. Prediction response contains confidence.
8. Clear functionality works on frontend.

For image preprocessing, verify:

```text
shape == (1, 28, 28)
```

==================================================
25. REQUIREMENTS.TXT
====================

Create an appropriate `requirements.txt`.

Include packages required by the implementation, such as:

```text
Flask
tensorflow
numpy
Pillow
gunicorn
python-dotenv
pytest
```

Use compatible versions rather than unnecessarily pinning random versions.

Make sure the application can be installed using:

```bash
pip install -r requirements.txt
```

==================================================
26. RUNNING LOCALLY
===================

The application must run with:

```bash
python run.py
```

or:

```bash
flask run
```

Prefer:

```bash
python run.py
```

for simplicity.

The README must explain:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

==================================================
27. GUNICORN
============

Prepare the application for production deployment.

Create a `Procfile` if useful:

```text
web: gunicorn run:app
```

If the Flask application factory pattern is used, adapt the Gunicorn command appropriately.

Do not use Flask's development server as the production server.

==================================================
28. DEPLOYMENT READINESS
========================

Prepare the application for deployment to a cloud platform that supports Python/Flask.

The README must explain:

1. Create repository.
2. Push code to GitHub.
3. Configure Python runtime.
4. Install requirements.
5. Ensure `digits_intel.h5` is available.
6. Configure environment variables.
7. Configure Gunicorn.
8. Deploy.
9. Test `/`.
10. Test `/api/predict`.

IMPORTANT:

The trained model file is required at runtime.

If the `.h5` file is too large for GitHub or the selected hosting provider, explain a suitable alternative such as model storage/object storage and loading it during deployment.

Do not silently omit the model.

==================================================
29. HEALTH CHECK
================

Create:

```text
GET /health
```

Response:

```json
{
    "status": "healthy",
    "model_loaded": true
}
```

Use this endpoint for deployment health checks.

==================================================
30. UI DETAILS
==============

Use polished UI details:

* subtle background grid
* soft radial glow
* glass-style cards
* rounded corners
* elegant typography
* smooth transitions
* hover states
* button micro-interactions
* animated result appearance
* animated probability bars
* clean icons

Do not over-animate.

Keep the interface fast and professional.

Use CSS variables for the design system.

Example:

```css
:root {
    --bg-primary: ...;
    --bg-secondary: ...;
    --accent-primary: ...;
    --accent-secondary: ...;
    --text-primary: ...;
    --text-secondary: ...;
    --border: ...;
}
```

Do not hard-code the same colors repeatedly throughout the stylesheet.

==================================================
31. BRANDING
============

Application branding:

**DigitAI**

Tagline:

**Intelligent Handwritten Digit Recognition**

Footer:

```text
DigitAI
Powered by TensorFlow & Flask
MNIST Neural Network
```

Add a small:

```text
Model: ANN • Input: 28×28 • Classes: 10
```

information section.

==================================================
32. ABOUT MODEL SECTION
=======================

Add a small professional information section.

Title:

**How DigitAI Works**

Show four steps:

```text
01
Upload
Upload a handwritten digit image.

02
Preprocess
Convert to grayscale and resize to 28×28.

03
Analyze
The trained neural network analyzes the image.

04
Predict
The model predicts one of 10 digits from 0–9.
```

This section should help users understand the ML pipeline.

==================================================
33. MODEL INFORMATION
=====================

Show:

```text
Model Architecture

Input
28 × 28 Grayscale

Hidden Layer 1
100 Neurons • ReLU

Hidden Layer 2
100 Neurons • ReLU

Output
10 Classes • Softmax
```

Keep this section visually clean.

==================================================
34. README
==========

Generate a professional README containing:

* Project title
* Project overview
* Features
* Tech stack
* ML architecture
* Preprocessing pipeline
* Folder structure
* Installation
* Running locally
* API documentation
* Testing
* Deployment
* Environment variables
* Troubleshooting
* Model file requirement
* Screenshots placeholder
* Future improvements

Explain the project in beginner-friendly language as well as technical language.

==================================================
35. CODE QUALITY
================

Follow professional coding standards.

Requirements:

* PEP 8
* meaningful variable names
* type hints where useful
* docstrings for important functions
* modular architecture
* no unnecessary duplicated code
* no hard-coded absolute Windows paths
* no notebook-specific code in production application
* no debugging print statements
* proper exception handling

IMPORTANT:

Do NOT copy this notebook code directly into Flask routes.

Instead, convert the notebook workflow into reusable production services.

==================================================
36. MODEL COMPATIBILITY CHECK
=============================

Before completing the application, inspect the supplied model file if available.

Verify:

* input shape
* output shape
* model architecture
* number of output classes
* expected preprocessing

The application should be compatible with the actual model.

If the model file is not present, do not invent its contents.

Use the architecture and preprocessing specified above as the expected model contract.

==================================================
37. IMAGE PREVIEW
=================

After upload:

Show:

```text
Original Image
```

with the uploaded image.

Optionally show:

```text
Model Input
28 × 28
```

as a small preview after preprocessing.

This helps demonstrate what the model actually receives.

Do not confuse the user by showing too many technical details.

==================================================
38. PREDICTION CONFIDENCE
=========================

Use the softmax probability corresponding to the predicted digit.

For example:

```python
probabilities = model.predict(image)[0]
digit = int(np.argmax(probabilities))
confidence = float(probabilities[digit])
```

Display:

```text
98.21%
```

Do not artificially modify confidence.

Do not round before calculations.

Only format the value for display.

==================================================
39. API ERROR HANDLING
======================

Use consistent API responses.

Success:

```json
{
    "success": true,
    "digit": 5,
    "confidence": 0.9734,
    "probabilities": {}
}
```

Failure:

```json
{
    "success": false,
    "error": "Human-readable error message"
}
```

Frontend must handle both.

==================================================
40. FINAL VALIDATION
====================

Before considering the project complete, verify:

[ ] Flask starts successfully.

[ ] Model loads successfully.

[ ] Home page opens.

[ ] Upload button works.

[ ] Drag and drop works.

[ ] Image preview works.

[ ] PNG works.

[ ] JPG works.

[ ] JPEG works.

[ ] WEBP works.

[ ] Invalid files are rejected.

[ ] Large files are rejected.

[ ] Prediction endpoint works.

[ ] Model receives shape `(1, 28, 28)`.

[ ] Predicted digit is between 0 and 9.

[ ] Confidence is displayed.

[ ] All 10 probabilities are displayed.

[ ] Loading state works.

[ ] Error state works.

[ ] Clear button works.

[ ] New image can be uploaded after clearing.

[ ] Mobile layout works.

[ ] `/health` works.

[ ] Tests pass.

[ ] No absolute local Windows paths exist.

[ ] No secret keys are committed.

[ ] Production configuration exists.

[ ] Gunicorn configuration exists.

[ ] README is complete.

==================================================
41. IMPORTANT ANTIGRAVITY EXECUTION INSTRUCTIONS
================================================

Do not merely generate a plan.

Actually create the complete project.

Create all required files and folders.

Implement the backend.

Implement the frontend.

Implement image preprocessing.

Implement model loading.

Implement prediction API.

Implement error handling.

Implement clear/reset functionality.

Implement responsive UI.

Implement tests.

Implement README.

After generating the project:

1. Inspect the complete project structure.
2. Check imports.
3. Check Flask routes.
4. Check model path.
5. Check image preprocessing.
6. Check frontend JavaScript.
7. Check API request/response format.
8. Check that the Clear button resets everything.
9. Run the application if the environment supports it.
10. Fix any errors found.
11. Do not leave placeholder functions where real implementation is expected.

If `digits_intel.h5` is unavailable in the workspace, clearly identify that as the only missing runtime artifact and still create the complete application around the specified model contract.

Do not replace the trained model with a mock model.

Do not create fake prediction results.

Do not use random numbers for confidence.

The final application must use the actual trained Keras model.

==================================================
42. FINAL DELIVERABLE
=====================

At the end, provide:

1. Complete project structure.
2. Explanation of important files.
3. How to run locally.
4. How to test.
5. How to deploy.
6. Required environment variables.
7. Model file location.
8. API endpoint documentation.
9. Any issue discovered during implementation.
10. Exact command to start the production server.

The final result should look like a real portfolio-ready AI product, not a basic tutorial project.

Prioritize:

**Correct ML preprocessing + reliable prediction + clean architecture + premium UI/UX + deployment readiness.**
