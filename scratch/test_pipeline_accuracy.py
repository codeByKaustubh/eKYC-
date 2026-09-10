import glob, cv2, numpy as np, pickle

with open('models/forensic_models.pkl', 'rb') as f:
    models = pickle.load(f)
clf = models['video_pipeline']
cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def evaluate_video(path):
    cap = cv2.VideoCapture(path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_f // 16)
    
    nf_list, nb_list, lap_f_list, lap_b_list, grad_list, fft_list, temp_list = [], [], [], [], [], [], []
    prev_g = None
    cached_box = None
    
    frame_idx = 0
    sampled = 0
    while cap.isOpened() and sampled < 16:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        sampled += 1
        
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if sampled % 4 == 1 or cached_box is None:
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = cascade.detectMultiScale(small, 1.15, 4, minSize=(30, 30))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                cached_box = (fx * 2, fy * 2, fw * 2, fh * 2)
            elif cached_box is None:
                cached_box = (int(w * 0.28), int(h * 0.18), int(w * 0.44), int(h * 0.58))
                
        bx, by, bw, bh = cached_box
        face = gray[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
        bg = gray[int(h * 0.75):int(h * 0.95), int(w * 0.05):int(w * 0.25)]
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
        
    cap.release()
    if not nf_list: return None
    
    m_nf = np.mean(nf_list)
    m_nb = np.mean(nb_list)
    m_lap_f = np.mean(lap_f_list)
    m_lap_b = np.mean(lap_b_list)
    
    feats = np.array([[
        m_nf,
        float(np.std(nf_list)),
        m_nb,
        float(m_nf / (m_nb + 1e-6)),
        m_lap_f,
        float(m_lap_f / (m_lap_b + 1e-6)),
        float(np.mean(grad_list)),
        float(np.mean(fft_list)),
        float(np.mean(temp_list)) if temp_list else 0.0
    ]])
    
    prob = float(clf.predict_proba(feats)[0][1])
    is_optically_natural = (m_nf >= 2.20 and m_lap_f >= 48.0)
    if is_optically_natural:
        is_attack = (prob >= 0.68)
    else:
        is_attack = (prob >= 0.55)
    return is_attack, prob, m_nf, m_lap_f

real_paths = sorted(glob.glob('Real/v*.mp4'))
fake_paths = sorted(glob.glob('Fake/vs*.mp4'))

real_passed = 0
for p in real_paths:
    res = evaluate_video(p)
    if res and not res[0]:
        real_passed += 1

fake_flagged = 0
for p in fake_paths:
    res = evaluate_video(p)
    if res and res[0]:
        fake_flagged += 1

print(f'=== FINAL VERIFICATION RESULTS ===')
print(f'Real Videos Verified: {real_passed}/{len(real_paths)} ({real_passed/len(real_paths)*100:.1f}%)')
print(f'Fake Videos Caught:   {fake_flagged}/{len(fake_paths)} ({fake_flagged/len(fake_paths)*100:.1f}%)')
