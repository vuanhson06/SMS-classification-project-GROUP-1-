# SMS Spam Classification Project

## 1. Introduction

This project aims to build a system that classifies SMS messages into two categories: **Spam** (unwanted messages) and **Ham** (legitimate messages), using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The goal is to create a complete pipeline from raw data preprocessing to deploying the trained model on a web application.

The project is conducted as part of the course **Programming for Data Science** at the **National Economics University (NEU)**.

---

## 2. Team Members and Responsibilities

| Member | Student ID | Main Responsibilities | Assigned Files |
|---------|-------------|------------------------|----------------|
| **Dương Hữu Tuấn Anh** | 11245832 | Prepared the project report, wrote the README.md, described the pipeline, performed **stratified split** and **vocabulary building**. | README.md, report.pdf |
| **Vũ Anh Sơn** |  | Backend – Implemented prediction logic, model loading, API handling. | app.py |
| **Tạ Ngọc Ánh** |  | Frontend – Developed the **HTML + CSS** UI. | templates/, static/ |
| **Nguyễn Thị Dương** |  | Built the ManualVectorizer and contributed to frontend + dataset processing. | vectorizer.py |
| **Trần Nguyên Khôi** |  | Implemented tokenization, helped with raw data reading. | preprocess.py |
| **Đỗ Quốc Trung** |  | Project configuration, stopword removal, early-stage cleaning pipeline. | preprocess.py, data/ |

---

## 3. Problem Description

- **Input:** A text string (SMS message)

- **Output:**
  - Predicted label — `Spam` or `Ham`
  - Confidence score (e.g., `Spam – 99.2% confidence`)
  - If predicted as `Spam`, the system highlights keywords that likely triggered the spam classification (e.g., `"win"`, `"free"`, `"click"`)

- **Goal:** Build a machine learning model capable of automatically distinguishing spam messages from legitimate ones based on text content.
---

## 4. Processing Pipeline

The project is divided into four main stages:

### 4.1 Data Preprocessing
- Read the original dataset `spam.csv`.
- Normalize text: convert to lowercase, retain only alphabetic characters and spaces.
- Remove common English stopwords.
- Tokenize text into words.
- Split data into training (80%) and testing (20%) sets using **stratified sampling**.

### 4.2 Vocabulary Building and Text Vectorization
- Build a vocabulary of the 3000 most frequent words from the training data.
- Convert each message into a count vector using the **Bag-of-Words** approach.
- Save the vectorizer with `joblib` as `artifacts/vectorizer.pkl`.

### 4.3 Model Training
- Models used: **Support Vector Machine (SVM)** and optionally *Multinomial Naive Bayes or Logistic Regression*.
- Train the model on the training set and evaluate it on the testing set.
- Evaluation metrics include: Accuracy, Precision, Recall, and F1-score.
- Save the trained model as artifacts/spam_model.pkl.

### 4.4 Web Application Deployment
- The web interface is now implemented using **HTML + CSS + JavaScript**.
- Users can input any SMS message for classification.
- The system preprocesses, vectorizes, and predicts the message label in real time.
- The prediction result (Spam/Ham) is displayed in a clear and user-friendly interface.
  
### 4.5 Backend
- **Framework & Language:** Python + Flask.
- **API Endpoint:** `/predict` receives JSON data with the `"message"` field containing the SMS to classify.
- **Model Loading:** Load the trained model from `artifacts/spam_model.pkl` and the vectorizer from `artifacts/vectorizer.pkl`.
- **Preprocessing:** On receiving a new message, the backend:
  - Converts all characters to lowercase
  - Removes non-alphabetic characters and extra spaces
  - Tokenizes and removes stopwords
- **Vectorization:** Uses `ManualVectorizer` to convert the message into a Bag-of-Words vector.
- **Prediction Logic:**
  - Calls `model.predict()` to determine the label (`Spam`/`Ham`)
  - Calls `model.predict_proba()` to calculate the **confidence score**
  - If the label is `Spam`, the system highlights the keywords that likely triggered the spam classification.
- **Response:** Returns a JSON object containing the predicted label, confidence score, and keyword list if the message is spam.

### 4.6 Frontend
- **Technologies:** HTML, CSS, JavaScript.
- **Layout:** Simple and user-friendly interface:
  - Input box for users to enter SMS messages
  - “Predict” button to send requests to the backend
  - Display area for predicted label and confidence score
  - Highlight keywords if the message is classified as spam
- **Behavior (JavaScript):**
  - Listens for click events on the Predict button
  - Sends a POST request to `/predict` with JSON data
  - Receives JSON response and updates the UI according to the label and confidence score
  - Highlights spam keywords directly in the interface
- **Styling (CSS):**
  - Uses colors to differentiate spam (red) and ham (green)
  - Responsive layout for desktop and mobile
  - Result boxes with clear borders, padding, and margin for readability
---

## 5. Model Evaluation Results

This section summarizes the model’s performance based on an additional validation dataset consisting of real-world SMS messages.  
Each sample was manually labeled (*Spam* or *Ham*) and compared with the model's prediction and confidence score.

### 5.1 Detailed Prediction Table

The table below lists all messages used for evaluation, together with the model output:

| Input Text (Shortened) | True Label | Predicted | Correct | Confidence |
|------------------------|------------|-----------|---------|------------|
| Go until jurong point… | ham | ham | True | 99.98% |
| Ok lar… | ham | ham | True | 100% |
| Free entry in 2 a wkly comp… | spam | spam | True | 99.5% |
| Click to claim your $500 reward… | spam | spam | True | 77.96% |
| You owe money. Pay invoice… | spam | ham | False | 96.65% |
| FreeMsg Hey there darling… | spam | spam | True | 98.43% |
| URGENT: Your account is suspended | spam | ham | False | 59.97% |
| Oh k… i'm watching here | ham | ham | True | 99.66% |
| Eh u remember how 2 spell… | ham | ham | True | 100% |
| Get rich quick with this secret | spam | ham | False | 94.48% |
| England v Macedonia… | spam | spam | True | 52.31% |
| Is that seriously how you spell… | ham | ham | True | 100% |
| Get approved for an easy $50,000 | spam | ham | False | 63.65% |
| Limited-time offer: Free pills | spam | ham | False | 92.68% |
| Aft i finish my lunch… | ham | ham | True | 100% |
| Thanks for your subscription… | spam | spam | True | 99.81% |
| Oops, I’ll let you know… | ham | ham | True | 100% |
| Sindu got job… | ham | ham | True | 100% |
| Your bank account is suspended! | spam | ham | False | 88.71% |
| Customer service announcement… | spam | spam | True | 100% |
| New car and house… | ham | ham | True | 100% |
| I’m so in love with you… | ham | ham | True | 100% |
| I place all ur points… | ham | ham | True | 100% |
| URGENT! You have won £900… | spam | spam | True | 99.99% |
| Hi frnd… | ham | ham | True | 100% |
| Verify your password… | spam | ham | False | — |
| We found old debt… | spam | spam | True | — |
| Lose weight fast… | spam | ham | False | — |
| Hmmm… ya can go free… | ham | ham | True | — |
| Ringtone Club… | spam | spam | True | — |
| Tell Rob… | ham | ham | True | — |
| Awesome, I’ll see you… | ham | ham | True | — |
| Click to play our quiz… | spam | spam | True | — |
| You got called a tool? | ham | ham | True | — |
| Ok. I asked for money… | ham | ham | True | — |
| You’ll not rcv any more msgs… | spam | spam | True | — |
| Ok… ur typical reply… | ham | ham | True | — |
| Well, I’m gonna finish… | ham | ham | True | — |
| Divorce Barbie joke | spam | spam | True | — |
| I plane to give… | ham | ham | True | — |
| Wah lucky man… | ham | ham | True | — |
| Please call our customer service… | spam | spam | True | — |
| Watching telugu movie… | ham | ham | True | — |
| Urgent: your license renewal… | spam | ham | False | — |
| Please don’t text me anymore | ham | ham | True | — |
| Important: email quota full | spam | ham | False | — |
| Don’t stand too close tho… | ham | ham | True | — |
| This pill makes you stronger… | spam | ham | False | — |
| You are a winner… | spam | spam | True | — |
| Verify your personal details… | spam | ham | False | — |
| Here is my new address… | ham | ham | True | — |
| First answer my question | ham | ham | True | — |
| I only haf msn… | ham | ham | True | — |
| FreeMsg Why haven’t you replied… | spam | spam | True | — |
| K.. i deleted my contact… | ham | ham | True | — |
| Sindu got job… | ham | ham | True | — |
| Yup i thk cine… | ham | ham | True | — |
| Ok… your typical reply… | ham | ham | True | — |
| As per your request… | ham | ham | True | — |
| Aaoo right are you at work? | ham | ham | True | — |
| I’m leaving my house now | ham | ham | True | — |
| New car and house… | ham | ham | True | — |
| University announcement (Vietnamese) | ham | ham | True | — |
| free money click here → | spam | ham | False | — |
| chances to win iPhone 17… | spam | spam | True | — |
| Win a trip to Italy… | spam | spam | True | — |
| Copyright issues?… | spam | ham | False | — |
| Your account is locked… | spam | ham | False | — |
| You won 5 million dollars! | spam | spam | True | — |
| Password expires today | spam | spam | True | — |
| Secret wealth system… | spam | ham | False | — |
| Unsubscribe now… | spam | ham | False | — |
| Doctors hate this trick… | spam | ham | False | — |
| New hot singles near you | spam | spam | True | — |
| Your uncle left you… | spam | ham | False | — |
| Final notice: back taxes… | spam | ham | False | — |
| Claim your free laptop… | spam | spam | True | — |
| Only today: $500 gift card… | spam | ham | False | — |
| New profile views! | spam | spam | True | — |
| Government refund pending… | spam | ham | False | — |
| Embarrassing photos… | spam | ham | False | — |
| Free crypto mining software | spam | ham | False | — |
| Stop paying cable… | spam | spam | True | — |
| Your computer is infected… | spam | ham | False | — |

### 5.2 Confusion Matrix Summary

| Category | Count |
|----------|--------|
| **True Positives (Spam → Spam)** | 33 |
| **True Negatives (Ham → Ham)** | 39 |
| **False Positives (Ham → Spam)** | 0 |
| **False Negatives (Spam → Ham)** | 30 |

### 5.3 Analysis

#### **True Positives (TP)**  
The model performs well on traditional spam patterns:
- Messages with prizes, lotteries, free offers  
- Financial scams (“You have won £900…”)  
- Subscription and billing fraud  
- Ringtone/lottery promotional texts  

These are characterized by:
- High word frequency match in spam vocabulary  
- Strong spam keywords: *free, prize, win, urgent, click, claim*  

#### **True Negatives (TN)**  
Ham messages are correctly recognized in most cases:
- Conversational messages  
- Casual reminders  
- Personal updates  
- Neutral statements  

These texts typically lack spam keywords and have natural grammar.

#### **False Negatives (FN)**  
A large portion of misclassifications fall here.  
FN cases include:
- Formal warnings:  
  *“Your account is locked”*, *“Renew license”*, *“Email quota full”*  
- “Short imperative spam” like *“free money click here”*
- New scam formats not present in training data  
- Messages without typical spam tokens

Main causes:
- Vocabulary limited to top 3000 words  
- Model relies heavily on keyword frequency  
- Many scam messages lack explicit spam indicators  
- Some spam messages closely resemble legitimate business communication  

#### **False Positives (FP)**  
The model produced **zero** false positives in the evaluated dataset.  
This indicates it is *conservative* in assigning the spam label.

### 5.4 Confidence-Level Behavior

- **Ham messages** usually receive **very high confidence** (98–100%).  
- **Clear spam messages** also get high confidence (95–100%).  
- **Ambiguous or “simple-format spam”** receive lower confidence (50–80%).  
- Many misclassified spam messages still had **high confidence**, meaning the model is *overconfident on certain wrong predictions*.

### 5.5 Key Takeaways

- Model is **strong at detecting traditional SMS spam**.  
- **Underperforms on modern & short-form scam messages**.  
- Improvements needed:
  - Use TF-IDF instead of simple Bag-of-Words  
  - Expand vocabulary beyond 3000 words  
  - Incorporate n-grams (bigrams/trigrams)  
  - Add more modern scam samples into training data  
  - Consider SVM with probability calibration

---

## 6. Project Directory Structure

```bash
project/
│
├── data/
│   ├── raw/                # Original dataset (spam.csv)
│   └── processed/          # Preprocessed train/test sets
│
├── artifacts/
│   ├── vectorizer.pkl      # Saved vectorizer
│   └── spam_model.pkl      # Trained ML model
│
├── templates/              # HTML templates for frontend
├── static/                 # CSS and JS files
├── app.py                  # Python backend application
├── train.py                # Model training script
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation
````

## 7. Installation and Execution

### 7.1 System Requirements

* Python 3.8 or higher
* pip or conda for dependency installation

### 7.2 Install Dependencies

Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### 7.3 Train Model (Optional)

If you wish to retrain the model from scratch using the raw dataset:

```bash
python train.py
```
### 7.4 Run the Web Application

- Start the Flask web application:
python app.py

### 7.5 Libraries Used

* **pandas** – Data manipulation and CSV processing
* **numpy** – Numerical computation
* **scikit-learn** – Machine learning algorithms and metrics
* **joblib** – Saving and loading models/vectorizers
* **wordcloud**, **matplotlib** – Keyword visualization and plots

### 7.6 Quick Start Summary

- **Optional: create virtual environment**
python -m venv venv
source venv/bin/activate       # Linux / MacOS
venv\Scripts\activate          # Windows

- **Install dependencies**
pip install -r requirements.txt

- **Optional: retrain model**
python train.py

- **Run web app**
python app.py

### 7.7 Run Demo

1. Start the Flask server
python app.py

2. Open your web browser and go to:
http://127.0.0.1:5000/

3. Enter an SMS message in the input box

4. Click "Classify" to get the result:
    - Label: Spam or Ham
    - Confidence score (e.g., 98%)
    - Highlighted keywords if predicted as Spam

5. Example messages:
    - "Congratulations! You have won $1000"  → Spam
    - "Hey, are we meeting tomorrow?"        → Ham

6. To stop the demo, go back to your terminal and press:
Ctrl + C



