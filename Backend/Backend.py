from fastapi import FastAPI, HTTPException, UploadFile, File
# FastAPI: Framework web dùng để tạo API backend.
# HTTPException: Dùng để trả lỗi HTTP (vd: 404, 500) có mô tả chi tiết.
# UploadFile, File: Cho phép nhận file từ người dùng khi upload CSV.
from fastapi.middleware.cors import CORSMiddleware # Cho phép frontend (HTML/JS) gọi API từ domain khác (bật CORS).
from pydantic import BaseModel # Định nghĩa cấu trúc dữ liệu vào/ra API
import joblib # Dùng để load model và vectorizer đã lưu
import numpy as np 
import pandas as pd
import os
import io
from typing import Dict, Any, List, Optional, Union
import logging # Ghi log khi server chạy, giúp debug.
from sklearn.base import BaseEstimator, TransformerMixin # Tạo custom vectorizer theo chuẩn scikit-learn.
import re

# MANUAL VECTORIZER: BỘ BIẾN ĐỔI (VECTORIZER) THỦ CÔNG, DÙNG ĐỂ CHUYỂN VĂN BẢN THÀNH VECTOR SỐ.
class ManualVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab=None):
        self.vocab = vocab or {} # vocab: danh sách các từ được dùng để huấn luyện mô hình
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)} # Tạo chỉ mục (index) cho từng từ trong vocab
    
    def fit(self, X, y=None):# Hàm fit không làm gì vì vocab đã có sẵn (fit đã được làm trong bước tiền xử lý)
        return self
    
    def transform(self, X):
        if isinstance(X, str): # Nếu chỉ truyền vào 1 chuỗi, đưa vào danh sách
            X = [X]
        
        results = []
        for text in X:
            features = np.zeros(len(self.vocab))# Tạo vector độ dài = số từ trong vocab, khởi tạo bằng 0
            words = text.lower().split()# tách câu thành danh sách từ (ở đây tách đơn giản theo khoảng trắng, lowercase).
            for word in words:
                if word in self.vocab_index: # Nếu từ nằm trong vocab thì tăng tần suất
                    features[self.vocab_index[word]] += 1
            results.append(features)
        
        return np.array(results)
    
    def fit_transform(self, X, y=None):# type: ignore  # không cần fit nữa vì đã tự build Vocab 
        return self.transform(X)# hàm biến đổi dữ liệu đầu vào (văn bản) thành dạng số học mà mô hình máy học có thể hiểu được.

    @property
    def vocabulary_(self):
        return self.vocab_index# Trả về từ điển vocab_index (phù hợp với chuẩn sklearn)

# CẤU HÌNH LOGGING CHO SERVER
logging.basicConfig(level=logging.INFO)# logging dùng để in log hoạt động ra console.
logger = logging.getLogger(__name__)

ARTIFACT_DIR = "artifacts" # Đường dẫn đến thư mục chứa mô hình và vectorizer
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")# đường dẫn đầy đủ tới vectorizer.pkl.
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.pkl")# đường dẫn tới mô hình tốt nhất (spam_model.pkl)

# KHỞI TẠO FASTAPI 
app = FastAPI(
    title="AmongSMS - Spam Detection API",
    description="API phát hiện tin nhắn rác (spam)",
    version="1.0.0"
)

# CẤU HÌNH CORS
# Cho phép giao diện web (frontend) ở bất kỳ nguồn nào có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins trong development
    allow_credentials=True,
    allow_methods=["*"],# Cho phép tất cả các loại HTTP method
    allow_headers=["*"],# Cho phép tất cả headers
)

# ĐỊNH NGHĨA KIỂU DỮ LIỆU  
class SMSRequest(BaseModel):
    text: str # Nội dung tin nhắn đầu vào

class SMSResponse(BaseModel):
    label: str #  Kết quả dự đoán: "spam" hoặc "ham"
    prob: Optional[float] # Xác suất là spam
    top_words: List[List[Union[str, int]]] # Từ khóa xuất hiện nhiều nhất
    confidence: float # Mức độ tin cậy (%)

class BatchResponse(BaseModel): # output cho batch prediction
    filename: str
    total_messages: int
    spam_count: int
    ham_count: int
    results: List[Dict]

# CORE FUNCTIONS 
vectorizer = None
model = None

def load_artifacts():
    global vectorizer, model
    if vectorizer is None or model is None:
        if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):# Nếu file ko tồn tại
            raise FileNotFoundError("Không tìm thấy vectorizer.pkl hoặc spam_model.pkl trong artifacts")
        vectorizer = joblib.load(VEC_PATH) # dùng joblib load vectorizer
        model_data = joblib.load(MODEL_PATH) # dùng joblib load model tốt nhất
        model = model_data['model'] if isinstance(model_data, dict) else model_data #Nếu model_data là dictionary, thì ta chỉ lấy phần 'model' trong đó. Ngược lại, nếu nó không phải dictionary (tức là bản thân joblib.load() trả về mô hình luôn) → ta gán trực tiếp.
        print(f"Model loaded: {type(model).__name__}") #In ra tên lớp của mô hình vừa load
        print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}") # In ra kích thước vocab của vectorizer
    return vectorizer, model

def get_feature_names(vec) -> np.ndarray:
    if hasattr(vec, "get_feature_names_out"): #Nếu vectorizer có method chuẩn của sklearn: get_feature_names_out()
        return np.array(vec.get_feature_names_out())

    vocab = getattr(vec, "vocabulary_", None) #Nếu không có get_feature_names_out, ta thử lấy thuộc tính vocabulary_
    if vocab is None:
        raise ValueError("Vectorizer không có vocabulary_ hoặc get_feature_names_out")
    
    # vocabulary_ thường là dict: {token: index}. Ta cần đảo lại thành mảng theo index.
    inv = [None] * len(vocab)
    for token, idx in vocab.items():# Duyệt từng (token, idx) trong vocab và gán token vào đúng vị trí idx
        inv[idx] = token
    return np.array(inv)


def extract_top_spam_words_fallback(text: str, top_k: int = 5):
    try:
        # Danh sách từ spam phổ biến với trọng số
        spam_keywords = {
            'free': 10, 'win': 9, 'won': 9, 'prize': 8, 'cash': 8, 'congratulations': 10,
            'claim': 7, 'urgent': 7, 'limited': 6, 'guaranteed': 6, 'click': 7, 'award': 6,
            'reward': 6, 'bonus': 5, 'discount': 5, 'offer': 5, 'deal': 4, 'sale': 4,
            'selected': 5, 'lucky': 5, 'winner': 7, 'million': 6, 'dollar': 5, 'money': 5,
            'credit': 4, 'loan': 4, 'text': 2, 'stop': 2, 'reply': 3, 'call': 3, 'now': 4,
            'today': 3, 'only': 3, 'special': 4, 'exclusive': 4, 'last': 3, 'chance': 4,
            'opportunity': 3, 'apply': 3, 'register': 3, 'sign': 3, 'subscribe': 3,
            'code': 3, 'password': 2, 'account': 2, 'premium': 6, 'subscription': 5,
            'membership': 4, 'buy': 4, 'purchase': 4, 'order': 4, 'price': 4, 'cost': 3,
            'payment': 3, 'card': 3, 'bank': 3, 'verify': 3, 'confirm': 3, 'access': 3,
            'unlock': 4, 'download': 3, 'mobile': 2, 'phone': 2, 'service': 2, 'gift': 5,
            'present': 4, 'extra': 3, 'clearance': 5, 'bargain': 4, 'promotion': 5,
            'trial': 4, 'new': 3, 'latest': 3, 'secret': 4, 'amazing': 4, 'awesome': 3,
            'best': 4, 'top': 4, 'quality': 3, 'luxury': 4, 'vip': 5, 'instant': 4,
            'quick': 3, 'easy': 3, 'profit': 5, 'income': 4, 'rich': 4, 'success': 4
        }
        
        words = text.lower().split() # Tách văn bản thành các từ riêng lẻ và chuyển về chữ thường
        found_spam_words = [] # Danh sách để lưu các từ spam tìm thấy
        
        # Duyệt qua từng từ trong tin nhắn để tìm từ spam
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower()) # Làm sạch từ: loại bỏ ký tự đặc biệt, chỉ giữ chữ cái và số
            if clean_word and len(clean_word) > 2:  # Chỉ xét từ có ít nhất 3 ký tự
                if clean_word in spam_keywords:
                    score = spam_keywords[clean_word] # Lấy trọng số spam của từ này
                    found_spam_words.append([clean_word, score]) # Thêm từ và trọng số vào danh sách kết quả
        
        # Loại bỏ các từ trùng lặp, giữ lại bản có trọng số cao nhất
        unique_words = {}
        for word, score in found_spam_words:
            if word not in unique_words or score > unique_words[word]:# Nếu từ chưa có trong dict, hoặc có trọng số cao hơn thì cập nhật
                unique_words[word] = score
        
        sorted_words = sorted(unique_words.items(), key=lambda x: x[1], reverse=True) # Sắp xếp các từ theo trọng số giảm dần
        result = [[word, score] for word, score in sorted_words[:top_k]]  # Chỉ lấy top_k từ quan trọng nhất
        
        # FALLBACK: nếu không tìm thấy từ spam, phân tích thêm
        if not result:
            # Thử tìm các từ có vẻ spam-like
            for word in words:# Duyệt lại qua các từ để tìm từ có vẻ spam-like dựa trên pattern
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if len(clean_word) > 3:# Chỉ xét từ có ít nhất 4 ký tự
                    # Kiểm tra các pattern spam thông thường
                    if any(pattern in clean_word for pattern in ['free', 'win', 'cash', 'prize', 'offer']):
                        result.append([clean_word, 3]) # Thêm từ này với trọng số mặc định thấp
                        if len(result) >= top_k:# Dừng khi đã đủ số lượng từ cần tìm
                            break
        
        print(f"Spam words from '{text[:30]}...': {result}")
        return result
        
    except Exception as e:
        print(f"Error in extract_top_spam_words: {e}")
        return [['spam', 5]]  # Fallback an toàn

def extract_top_spam_words(text: str, top_k: int = 5):
    try:
        vec, clf = load_artifacts()
        feature_names = get_feature_names(vec)

        X = vec.transform([text])
        x_row = X[0]

        importance = None
        if hasattr(clf, "coef_"):
            classes = getattr(clf, "classes_", np.array([0, 1]))
            if 1 in classes:
                spam_idx = int(np.where(classes == 1)[0][0])
            elif "spam" in classes:
                spam_idx = int(np.where(classes == "spam")[0][0])
            else:
                spam_idx = 0
            importance = clf.coef_[spam_idx]
        elif hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_

        if importance is None:
            print("Model không có coef_ / feature_importances_, dùng fallback.")
            return extract_top_spam_words_fallback(text, top_k=top_k)

        contrib = {}

        if hasattr(x_row, "tocoo"):
            x_coo = x_row.tocoo()
            for idx, value in zip(x_coo.col, x_coo.data):
                score = float(value * importance[idx])
                if score > 0:
                    token = feature_names[idx]
                    contrib[token] = contrib.get(token, 0.0) + score
        else:
            x_arr = np.asarray(x_row).ravel()
            for idx, value in enumerate(x_arr):
                if value == 0:
                    continue
                score = float(value * importance[idx])
                if score > 0:
                    token = feature_names[idx]
                    contrib[token] = contrib.get(token, 0.0) + score

        if not contrib:
            return extract_top_spam_words_fallback(text, top_k=top_k)

        sorted_words = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = [[w, float(s)] for w, s in sorted_words]

        print(f"Spam words (model-based) from '{text[:30]}...': {result}")
        return result

    except Exception as e:
        print(f"Error in extract_top_spam_words (model-based): {e}")
        return extract_top_spam_words_fallback(text, top_k=top_k)

def predict_one(text: str): # Hàm dự đoán cho một tin nhắn đơn lẻ
    try:
        vec, model = load_artifacts()
        X = vec.transform([text]) # Vector hóa

        # Dự đoán nhãn
        y_pred = model.predict(X)[0] # dùng best model đã train và hàm predict() của skikit-learn -> 1 or 0
        label = "spam" if y_pred == 1 else "ham"

        print(f"Prediction: {label} for text: {text[:50]}...")

        # Tính xác suất và độ tin cậy
        confidence = 85.0
        
        if hasattr(model, "predict_proba"): #kiểm tra xem model có hàm predict_proba không 
            proba = model.predict_proba(X)[0] # tính probability của spam và ham
            prob = float(proba[1])
            confidence = round(prob * 100, 2) if label == "spam" else round((1 - prob) * 100, 2)
            print(f"Confidence: {confidence}%")
        else:# Nếu model không hỗ trợ xác suất
            prob = 0.85 if label == "spam" else 0.15
            print(f"Using fallback confidence: {confidence}%")

        # Trích xuất top 10 từ gây spam trong tin nhắn
        top_words = []
        if label == "spam":
            top_words = extract_top_spam_words(text)
            print(f"Top spam words: {top_words}")
        
        return {
            "label": label,
            "prob": prob,
            "top_words": top_words,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Lỗi predict_one: {e}")
        raise e

# ENDPOINTS
# Endpoint kiểm tra nhanh API có hoạt động hay không
@app.get("/")
async def root():
    return {"message": "AmongSMS Spam Detection API", "status": "running"}

#Endpoint Kiểm tra trạng thái tải mô hình và vectorizer
@app.get("/health")
async def health_check():
    try:
        vec, model = load_artifacts()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Endpoint /predict: dự đoán một tin nhắn
@app.post("/predict", response_model=SMSResponse)
async def predict_sms(request: SMSRequest):
    try:
        print(f"Received prediction request: {request.text[:100]}...")
        result = predict_one(request.text)
        print(f"Prediction result: {result}")
        return SMSResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Lỗi khi dự đoán: {str(e)}")

# Endpoint xử lý file CSV (trả kết quả dạng JSON)
@app.post("/batch-predict-json")# Người dùng sẽ upload file CSV qua endpoint này.
async def batch_predict_json(file: UploadFile = File(...)): # Nhận file upload từ client
    try:
        print(f"Processing batch file: {file.filename}")
        
        content = await file.read() # Đọc toàn bộ nội dung file từ client upload
        
        # Thử các encoding khác nhau để đọc file CSV
        encodings = ['utf-8', 'latin-1', 'utf-16', 'windows-1252', 'cp1252']
        df = None # Khởi tạo biến lưu DataFrame
        
        for encoding in encodings:
            try:
                csv_content = content.decode(encoding) # Giải mã nội dung file với encoding hiện tại
                df = pd.read_csv(io.StringIO(csv_content)) # Đọc nội dung CSV vào DataFrame
                print(f"Successfully decoded with {encoding}")
                break
            except Exception as e:
                print(f"Failed with {encoding}: {e}")
                continue
        
        if df is None: # Fallback: nếu không decode được, để pandas tự detect encoding
            try:
                df = pd.read_csv(io.BytesIO(content)) # Đọc trực tiếp từ bytes mà không decode trước
                print("Successfully read with pandas auto-detection")
            except Exception as e:
                raise Exception(f"Không thể đọc file CSV: {str(e)}")
        
        print(f" CSV columns: {list(df.columns)}")
        print(f" Sample data: {df.head(2)}")
        
        # Tìm cột text linh hoạt - hỗ trợ nhiều tên cột khác nhau
        text_column = None
        possible_columns = ['text', 'message', 'sms', 'content', 'body', 'Message', 'SMS', 'Text']
        
        for col in possible_columns: # Duyệt qua danh sách tên cột có thể có
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None and len(df.columns) > 0: # Fallback: nếu không tìm thấy cột theo tên, dùng cột đầu tiên
            text_column = df.columns[0]
        
        # Kiểm tra xem đã tìm thấy cột text chưa
        if text_column is None:
            raise Exception("Không tìm thấy cột chứa tin nhắn trong file CSV")
            
        print(f"Using column: {text_column}")
        
        # Lấy texts và lọc bỏ giá trị NaN, rỗng
        texts = df[text_column].astype(str).tolist() # Chuyển cột text thành list
        texts = [text.strip() for text in texts if text and text.lower() != 'nan' and text.strip()]# Loại bỏ giá trị: None, 'nan', và chuỗi chỉ toàn khoảng trắng
        
        print(f" Processing {len(texts)} messages")
        
        if len(texts) == 0: # Kiểm tra xem còn tin nhắn nào để xử lý không
            raise Exception("Không có tin nhắn nào để xử lý")
        
        # Load model và predict
        vec, model = load_artifacts()
        X = vec.transform(texts)
        preds = model.predict(X)
        
        # Tạo kết quả
        results = []
        spam_count = 0
        
        for i, text in enumerate(texts): # Duyệt qua từng tin nhắn để xử lý chi tiết
            is_spam = bool(preds[i] == 1)  # Chuyển prediction thành boolean

            if is_spam:
                spam_count += 1
            
            # Tính confidence
            confidence = 85.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X[i:i+1])[0]
                confidence = round(proba[1] * 100, 2) if is_spam else round(proba[0] * 100, 2)
            
            # Luôn trích xuất từ spam khi là spam
            top_spam_words = extract_top_spam_words(text) if is_spam else [] # Chỉ extract từ spam cho tin nhắn spam để tiết kiệm tính toán
            
            results.append({
                "id": i + 1,
                "text": text,
                "predicted_label": "spam" if is_spam else "ham",
                "is_spam": is_spam,
                "top_spam_words": top_spam_words,
                "confidence": confidence
            })
        
        # Tổng hợp kết quả toàn batch
        response_data = {
            "filename": file.filename,
            "total_messages": len(results),
            "spam_count": spam_count,
            "ham_count": len(results) - spam_count,
            "spam_rate": round((spam_count / len(results)) * 100, 2) if len(results) > 0 else 0,
            "results": results,
            "success": True
        }
        
        print(f" Batch processing completed: {spam_count}/{len(results)} spam")
        return response_data
        
    except Exception as e:
        print(f" Batch error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn # là web server ASGI (Asynchronous Server Gateway Interface).Đây là server thật sự chạy ứng dụng FastAPI
    # CHẠY PORT 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
