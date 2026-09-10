import glob, cv2, numpy as np, pandas as pd

cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def get_feats(path):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(12):
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
    return {
        'nf': np.mean(nf_list),
        'nb': np.mean(nb_list),
        'ratio': np.mean(nf_list)/(np.mean(nb_list)+1e-6),
        'lap_f': np.mean(lap_f_list),
        'lap_ratio': np.mean(lap_f_list)/(np.mean(lap_b_list)+1e-6),
        'grad': np.mean(grad_list),
        'fft': np.mean(fft_list),
        'temp': np.mean(temp_list) if temp_list else 0.0
    }

reals = [get_feats(p) for p in glob.glob('Real/v*.mp4')[:30]]
fakes = [get_feats(p) for p in glob.glob('Fake/vs*.mp4')[:30]]

df_r = pd.DataFrame([r for r in reals if r])
df_f = pd.DataFrame([f for f in fakes if f])

print('=== REAL STATS ===')
print(df_r.describe().loc[['mean', 'std', 'min', '50%', 'max']])
print('\n=== FAKE STATS ===')
print(df_f.describe().loc[['mean', 'std', 'min', '50%', 'max']])
