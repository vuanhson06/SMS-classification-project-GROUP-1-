# SMS Spam Classification Project

## 1. Introduction

This project aims to build a system that classifies SMS messages into two categories: **Spam** (unwanted messages) and **Ham** (legitimate messages), using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The goal is to create a complete pipeline from raw data preprocessing to deploying the trained model on a web application.

The project is conducted as part of the course **Programming for Data Science** at the **National Economics University (NEU)**.

---

## 2. Team Members and Responsibilities

| Member | Student ID | Main Responsibilities | Assigned Files | Contribution |
|---------|-------------|------------------------|----------------|--------------|
| **Dương Hữu Tuấn Anh** | 11245832 | Wrote the README.md, described the pipeline, performed **stratified split** and **vocabulary building**. | README.md, split_train_test.py |  |
| **Vũ Anh Sơn** | 11245930 | Backend – Implemented prediction logic, model loading, API handling. | Backend.py, train_model.py |  |
| **Tạ Ngọc Ánh** | 11245844 | Frontend – Developed the **HTML + CSS** UI. | Frontend |  |
| **Nguyễn Thị Dương** | 11245866 | Built the ManualVectorizer and contributed to frontend + dataset processing. | Vectorize.py |  |
| **Trần Nguyên Khôi** | 11245889 | Implemented tokenization, helped with raw data reading, preparing slides. | Tokenize.py |  |
| **Đỗ Quốc Trung** | 11245944 | Project configuration, stopword removal, early-stage cleaning pipeline, testing web. | clean_stop_words.py |  |

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

### 4.1 Data Preprocessing & Vectorization
- Load the original dataset `spam.csv`.  
- Normalize text: convert to lowercase and keep only alphabetic characters and spaces.  
- Remove common English stopwords and tokenize each message into words.  
- Build a vocabulary of the **3000 most frequent words** from the training data.  
- Convert messages into numerical vectors using the **Bag-of-Words** model.  
- Split the dataset into training (80%) and testing (20%) sets using **stratified sampling**.  
- Save the trained vectorizer using `joblib` as `artifacts/vectorizer.pkl`.

### 4.2 Model Training
- **Models used:** SVM, Multinomial Naive Bayes (MNB), Logistic Regression (LR).
- Train all models on the training set and evaluate on the testing set using Accuracy, Precision, Recall, and F1-score.
- Select the best model based on weighted F1-score. **Logistic Regression (LR)** is chosen.
- Save the best model as `artifacts/spam_model.pkl`.

### 4.3 Web Application Deployment
- The web interface is now implemented using **HTML + CSS + JavaScript**.
- Users can input any SMS message for classification.
- The system preprocesses, vectorizes, and predicts the message label in real time.
- The prediction result (Spam/Ham) is displayed in a clear and user-friendly interface.
  
### 4.4 Backend
- **Framework & Language:** Python + FlaskAPI.
- **API Endpoints:**
  - `/predict` receives JSON data with the `"message"` field containing a single SMS to classify.
  - `/batch_predict` receives JSON data with the `"messages"` field containing a list of SMS messages for batch classification.
- **Model Loading:** Load the trained model from `artifacts/spam_model.pkl` and the vectorizer from `artifacts/vectorizer.pkl`.
- **Preprocessing:** On receiving a new message, the backend:
  - Converts all characters to lowercase
  - Removes non-alphabetic characters and extra spaces
  - Tokenizes and removes stopwords
- **Vectorization:** Uses `ManualVectorizer` to convert the message(s) into Bag-of-Words vector(s).
- **Prediction Logic:**
  - For single message: 
    - Calls `model.predict()` to determine the label (`Spam`/`Ham`)
    - Calls `model.predict_proba()` to calculate the **confidence score**
    - If the label is `Spam`, highlights keywords that likely triggered the spam classification
  - For batch messages: 
    - Applies the same preprocessing, vectorization, and prediction steps to each message
    - Returns predictions and confidence scores for all messages
- **Response:** Returns a JSON object containing the predicted label, confidence score, and keyword list if the message is spam. For batch prediction, returns a list of results for each message.


### 4.5 Frontend
- **Technologies:** HTML, CSS, JavaScript.
- **Layout:** Simple and user-friendly interface with two tabs:
  - **Single Prediction Tab:**
    - Input box for users to enter an SMS message
    - “Analyze Message” button to send a single prediction request to the backend
    - Display area for predicted label and confidence score
    - Highlight keywords if the message is classified as spam
  - **Batch Upload Tab:**
    - File upload or text area to enter multiple SMS messages
    - “Process Batch” button to send batch prediction requests to the backend
    - Display area showing predicted labels and confidence scores for all messages
- **Behavior (JavaScript):**
  - Listens for click events on the “Analyze Message” and “Process Batch” buttons
  - Sends POST requests to `/predict` (single) or `/batch_predict` (batch) with JSON data
  - Receives JSON response and updates the UI according to labels and confidence scores
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

| Input Text | Spam or Ham | True/False | Confidence Level |
|------------|-------------|------------|-----------------|
| Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat... | ham | True | 99.98% |
| Ok lar... Joking wif u oni... | ham | True | 100% |
| Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's | spam | True | 99.5% |
| Click to claim your $500 reward now! | spam | True | 77.96% |
| You owe money. Pay this invoice immediately. | spam | False | 96.65% |
| FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, å£1.50 to rcv | spam | True | 98.43% |
| Urgent: Your account is now suspended. | spam | False | 59.97% |
| Oh k...i'm watching here:) | ham | True | 99.66% |
| Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet. | ham | True | 100% |
| Get rich quick with this amazing secret. | spam | False | 94.48% |
| England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+ | spam | True | 52.31% |
| Is that seriously how you spell his name? | ham | True | 100% |
| Get approved for an easy $ 50,000 today, free of payment! | spam | False | 63.65% |
| Limited-time offer: Free medical pills. | spam | False | 92.68% |
| Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already? | ham | True | 100% |
| Thanks for your subscription to Ringtone UK your mobile will be charged å£5/month Please confirm by replying YES or NO. If you reply NO you will not be charged | spam | True | 99.81% |
| Oops, I'll let you know when my roommate's done | ham | True | 100% |
| Sindu got job in birla soft .. | ham | True | 100% |
| your bank account is suspended! Click here to get the information back | spam | False | 88.71% |
| Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery | spam | True | 100% |
| New car and house for my parents.:)i have only new job in hand:) | ham | True | 100% |
| I'm so in love with you. I'm excited each day i spend with you. You make me so happy. | ham | True | 100% |
| I place all ur points on e cultures module already. | ham | True | 100% |
| URGENT! We are trying to contact you. Last weekends draw shows that you have won a å£900 prize GUARANTEED. Call 09061701939. Claim code S89. Valid 12hrs only | spam | True | 99.99% |
| Hi frnd, which is best way to avoid missunderstding wit our beloved one's? | ham | True | 100% |
| Verify your password or account deletion, dont miss this! | ham | True | 95.82% |
| We found old debt; settle it now for a discount. | spam | True | 87.25% |
| Lose weight fast! Guaranteed results. Limited time offer only! | spam | False | 94.42% |
| Hmmm.. Thk sure got time to hop ard... Ya, can go 4 free abt... Muz call u to discuss liao... | ham | True | 94.86% |
| Ringtone Club: Get the UK singles chart on your mobile each week and choose any top quality ringtone! This message is free of charge. | spam | True | 99.99% |
| Tell rob to mack his gf in the theater | ham | True | 100% |
| Awesome, I'll see you in a bit | ham | True | 98.92% |
| click to play our quiz and win a trip to Russia | spam | True | 97.56% |
| You got called a tool? | ham | True | 98.81% |
| Ok. I asked for money how far | ham | True | 99.08% |
| You'll not rcv any more msgs from the chat svc. For FREE Hardcore services text GO to: 69988 If u get nothing u must Age Verify with yr network & try again | spam | True | 99.86% |
| Ok... Ur typical reply... | ham | True | 94.18% |
| Well, i'm gonna finish my bath now. Have a good...fine night. | ham | True | 99.72% |
| Did you hear about the new \"Divorce Barbie\"? It comes with all of Ken's stuff! | spam | True | 88.67% |
| I plane to give on this month end | ham | True | 97.92% |
| Wah lucky man... Then can save money... Hee... | ham | True | 99.23% |
| Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or å£5000 prize! | spam | True | 100% |
| Watching telugu movie..wat abt u? | ham | True | 98.87% |
| Urgent: Your license renewal requires immediate action. | spam | False | 89.9% |

### 5.2 Confusion Matrix Summary

| Metric              | Value |
|--------------------|-------|
| Total examples      | 267   |
| Spam messages       | 150   |
| Ham messages        | 117   |
| True predictions    | 234   |
| False predictions   | 33    |
| Model accuracy      | ~87.6%|

*Additional note:* We tried multiple models and various parameter settings, and this is the best result we achieved.

### 5.4 Confidence-Level Behavior

- **Ham messages** generally receive **very high confidence** (≈95–100%).  
- **Obvious spam messages** also get high confidence (≈90–100%).  
- **Ambiguous or “simple-format spam” messages** receive lower confidence (≈50–80%).  
- Some misclassified spam messages still had **moderate-to-high confidence**, indicating the model is sometimes *overconfident on wrong predictions*.  
- The confidence distribution aligns with message clarity: clearer messages → higher confidence, ambiguous messages → lower confidence.
---

## 6. Project Directory Structure

```bash
project/
│
├── Backend/
│   └── Backend.py                # Python backend application
│
├── Data Preprocessing/
│   ├── Tokenize.py               # Tokenization script
│   ├── Vectorize.py              # Vectorization script
│   ├── clean_stop_words.py       # Stop words cleaning script
│   └── split_train_test.py       # Train/test splitting script
│
├── Frontend/
│   ├── app.js                     # Frontend JS
│   ├── index.html                 # Main HTML file
│   ├── style.css                  # CSS styling
│   ├── image.png                  # Example images
│   ├── crewmate.png
│   └── crewmate-batch.png
│
├── Model Training/
│   └── train_model.py             # Script to train ML model
│
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # Preprocessed train/test sets
│
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
├── __pycache__/                   # Auto-generated Python cache files
│   ├── Backend.cpython-313.pyc
│   ├── Tokenize.cpython-313.pyc
│   ├── Vectorize.cpython-313.pyc
│   ├── csv.cpython-313.pyc
│   └── split_train_test.cpython-313.pyc
│
├── artifacts/                     # Trained models and vectorizer
│   ├── lr_model.pkl               # Logistic Regression model
│   ├── nb_model.pkl               # Naive Bayes model
│   ├── spam_model.pkl             # Final spam classification model
│   ├── svm_model.pkl              # Support Vector Machine model
│   ├── vectorizer.pkl             # Trained vectorizer
│   └── vocab.txt                  # Vocabulary list
│
├── data/                          # Input datasets
│   ├── raw/                       # Original dataset
│   │   └── spam.csv               # Raw spam email data
│   └── processed/                 # Preprocessed train/test sets
│
└── reports/                       # Model evaluation reports
    ├── best_model_summary.json    # Summary of best-performing model
    └── metrics.csv                # Evaluation metrics

```

## 7. Installation and Execution

### 7.1 System Requirements

* Python 3.8 or higher
* pip or conda for dependency installation

### 7.2 Install Dependencies
* STEP 1: SET UP ENVIRONMENT
```bash 
cd Project-Py
```
 - Activate virtual environment (if available)
```bash
source .venv/bin/activate        # Mac / Linux
or
.venv\Scripts\activate           # Windows

- Install required dependencies
pip install -r "Data Preprocessing/requirements.txt"
````

* STEP 2: DATA PREPROCESSING
(Run the files in this exact order)
```bash
python "Data Preprocessing/Tokenize.py"
python "Data Preprocessing/clean_stop_words.py"
python "Data Preprocessing/Vectorize.py"
python "Data Preprocessing/split_train_test.py"
````

* STEP 3: TRAIN THE MODEL

```bash
python "Model Training/train_model.py"
```

* STEP 4: RUN THE APPLICATION
```bash
python Backend/Backend.py
```
* STEP 5: ACCESS THE APPLICATION

 - Open your browser and go to:
```bash
http://localhost:8000
```
