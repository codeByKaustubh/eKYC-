import glob, cv2, numpy as np, pickle, os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

data_cache = "scratch/features_cache.pkl"
if os.path.exists(data_cache):
    with open(data_cache, "rb") as f:
        X, y, paths = pickle.load(f)
else:
    cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    def extract_face_forensics(path):
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(15):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        if not frames: return None
        
        nf_list, nb_list, lap_f_list, lap_b_list, grad_list, fft_list, temp_list = [], [], [], [], [], [], []
        prev_g = None
        for frame in frames:
            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = cascade.detectMultiScale(small, 1.15, 4, minSize=(30, 30))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                bx, by, bw, bh = fx*2, fy*2, fw*2, fh*2
            else:
                bx, by, bw, bh = int(w*0.28), int(h*0.18), int(w*0.44), int(h*0.58)
                
            face = gray[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
            bg = gray[int(h*0.75):int(h*0.95), int(w*0.05):int(w*0.25)]
            if face.size == 0 or bg.size == 0: continue
            
            nf = float(np.var(cv2.absdiff(face, cv2.medianBlur(face, 3))))
            nb = float(np.var(cv2.absdiff(bg, cv2.medianBlur(bg, 3))))
            lap_f = float(cv2.Laplacian(face, cv2.CV_64F).var())
            lap_b = float(cv2.Laplacian(bg, cv2.CV_64F).var())
            gx = cv2.Sobel(face, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(face, cv2.CV_64F, 0, 1, ksize=3)
            grad = float(np.mean(np.sqrt(gx**2 + gy**2)))
            f_thumb = cv2.resize(face, (64, 64))
            mag = np.abs(np.fft.fftshift(np.fft.fft2(f_thumb)))
            low = mag[24:40, 24:40]
            hf_ratio = float((np.sum(mag**2) - np.sum(low**2)) / (np.sum(mag**2) + 1e-9))
            if prev_g is not None and prev_g.shape == face.shape:
                temp_list.append(float(np.mean(cv2.absdiff(face, prev_g))))
            prev_g = face
            nf_list.append(nf)
            nb_list.append(nb)
            lap_f_list.append(lap_f)
            lap_b_list.append(lap_b)
            grad_list.append(grad)
            fft_list.append(hf_ratio)
            
        if not nf_list: return None
        m_nf = np.mean(nf_list)
        m_nb = np.mean(nb_list)
        m_lap_f = np.mean(lap_f_list)
        m_lap_b = np.mean(lap_b_list)
        return [
            m_nf,
            float(np.std(nf_list)),
            m_nb,
            float(m_nf / (m_nb + 1e-6)),
            m_lap_f,
            float(m_lap_f / (m_lap_b + 1e-6)),
            float(np.mean(grad_list)),
            float(np.mean(fft_list)),
            float(np.mean(temp_list)) if temp_list else 0.0
        ]

    X, y, paths = [], [], []
    for p in sorted(glob.glob("Real/v*.mp4")):
        f = extract_face_forensics(p)
        if f: X.append(f); y.append(0); paths.append(p)
    for p in sorted(glob.glob("Fake/vs*.mp4")):
        f = extract_face_forensics(p)
        if f: X.append(f); y.append(1); paths.append(p)
    X = np.array(X)
    y = np.array(y)
    with open(data_cache, "wb") as f:
        pickle.dump((X, y, paths), f)

print(f"Dataset: {len(X)} samples ({sum(y==0)} Real, {sum(y==1)} Fake)")

models = {
    "LogisticRegression": Pipeline([('s', RobustScaler()), ('clf', LogisticRegression(C=0.5, class_weight='balanced', random_state=42))]),
    "RandomForest(shallow)": Pipeline([('s', RobustScaler()), ('clf', RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42))]),
    "GradientBoosting(lr=0.05)": Pipeline([('s', RobustScaler()), ('clf', GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42))]),
    "ExtraTrees(max_depth=3)": Pipeline([('s', RobustScaler()), ('clf', ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42))])
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)

for name, model in models.items():
    preds = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, preds)
    for th in [0.40, 0.50, 0.55, 0.60]:
        bin_preds = (preds >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, bin_preds).ravel()
        r_acc = tn / (tn + fp) * 100
        f_acc = tp / (tp + fn) * 100
        print(f"[{name}] th={th:.2f} -> Real Correct: {r_acc:.1f}% ({tn}/{tn+fp}), Fake Correct: {f_acc:.1f}% ({tp}/{tp+fn}), AUC: {auc:.3f}")
