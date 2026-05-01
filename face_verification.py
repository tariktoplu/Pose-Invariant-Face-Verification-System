import cv2
import dlib
import numpy as np
import urllib.request
import shutil
from pathlib import Path
from skimage.feature import local_binary_pattern, hog
from skimage.measure import shannon_entropy
from scipy.spatial.distance import cosine

class ClassicalFaceVerifier:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        # 1. Face Detection & Landmark Detection
        self.detector = dlib.get_frontal_face_detector()
        
        # Fallback Classical Detectors for extreme occlusions / angles
        self.haar_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.haar_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.haar_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        # LBP cascade if available locally
        self.lbp_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        
        # Finer alignment cascades
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Build Gabor filter bank
        self.gabor_filters = self._build_gabor_filters()

        predictor_path = self._resolve_predictor_path(predictor_path)
        self.predictor = dlib.shape_predictor(str(predictor_path))
        
        # Canonical face definitions
        self.canonical_size = (200, 200)
        self.canonical_eyes = np.float32([(60, 60), (140, 60), (100, 140)]) # Left eye, Right eye, Mouth center
        
        # Define patches in canonical face: (x, y, w, h)
        # Genişletilmiş ve optimize edilmiş yama (patch) boyutları
        self.patches = {
            'left_eye': (30, 40, 60, 40),
            'right_eye': (110, 40, 60, 40),
            'nose': (70, 75, 60, 50),
            'mouth': (60, 130, 80, 40),
            'left_cheek': (15, 90, 50, 50),
            'right_cheek': (135, 90, 50, 50)
        }
        
        # For symmetric feature recovery
        self.symmetry_map = {
            'left_eye': 'right_eye',
            'right_eye': 'left_eye',
            'left_cheek': 'right_cheek',
            'right_cheek': 'left_cheek'
        }

    def _build_gabor_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 4):
            for lamda in [np.pi/4, np.pi/2, np.pi]:
                kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    def _resolve_predictor_path(self, predictor_path):
        predictor_file = Path(predictor_path)
        if predictor_file.exists() and predictor_file.stat().st_size > 0:
            return predictor_file

        try:
            import face_recognition_models

            bundled_path = Path(face_recognition_models.pose_predictor_model_location())
            if bundled_path.exists() and bundled_path.stat().st_size > 0:
                return bundled_path
        except Exception:
            pass

        print("Downloading dlib 68-point shape predictor...")
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = predictor_file.with_suffix(predictor_file.suffix + ".bz2")

        with urllib.request.urlopen(url) as response, open(compressed_path, "wb") as compressed_file:
            shutil.copyfileobj(response, compressed_file)

        import bz2

        with bz2.open(compressed_path, "rb") as compressed_file, open(predictor_file, "wb") as output_file:
            shutil.copyfileobj(compressed_file, output_file)

        compressed_path.unlink(missing_ok=True)
        print("Download complete.")
        return predictor_file

    def detect_and_align(self, image):
        """Detect face using ensemble classical methods, get landmarks, and align."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Image Enhancement for fallback
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        gray_eq = cv2.equalizeHist(gray)
        
        rect = None
        
        # 1. Primary: Dlib HOG (Original)
        rects = self.detector(gray, 1)
        if len(rects) > 0:
            rect = max(rects, key=lambda r: r.area())
            
        # 2. Fallback: Dlib HOG (CLAHE)
        if rect is None:
            rects = self.detector(gray_clahe, 1)
            if len(rects) > 0:
                rect = max(rects, key=lambda r: r.area())
                
        # Helper to convert OpenCV rect to Dlib rect
        def cv2_to_dlib(faces):
            if len(faces) == 0:
                return None
            # Get largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            return dlib.rectangle(left=int(x), top=int(y), right=int(x+w), bottom=int(y+h))

        # 3. Fallback: Haar Default (Equalized)
        if rect is None:
            faces = self.haar_default.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            rect = cv2_to_dlib(faces)
            
        # 4. Fallback: Haar Profile (Original) - for faces turned away
        if rect is None:
            faces = self.haar_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            rect = cv2_to_dlib(faces)

        # 5. Fallback: LBP (Original) - highly robust to lighting
        if rect is None and not self.lbp_cascade.empty():
            faces = self.lbp_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            rect = cv2_to_dlib(faces)
            
        if rect is None:
            return None, None, None
            
        # Get landmarks from the found rectangle
        shape = self.predictor(gray, rect)
        
        # Extract eye and mouth coordinates
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            
        left_eye = coords[36:42].mean(axis=0)
        right_eye = coords[42:48].mean(axis=0)
        mouth = coords[48:68].mean(axis=0)
        
        # Finer alignment: Search for eyes in the local neighborhood
        def refine_eye(pt, gray_img, cascade, window=30):
            x, y = int(pt[0]), int(pt[1])
            x1, y1 = max(0, x - window), max(0, y - window)
            x2, y2 = min(gray_img.shape[1], x + window), min(gray_img.shape[0], y + window)
            roi = gray_img[y1:y2, x1:x2]
            if roi.size > 0 and not cascade.empty():
                eyes = cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
                if len(eyes) > 0:
                    ex, ey, ew, eh = eyes[0]
                    return np.array([x1 + ex + ew/2.0, y1 + ey + eh/2.0])
            return pt

        left_eye = refine_eye(left_eye, gray, self.eye_cascade)
        right_eye = refine_eye(right_eye, gray, self.eye_cascade)
        
        src_pts = np.float32([left_eye, right_eye, mouth])
        
        # Affine transformation matrix
        M = cv2.getAffineTransform(src_pts, self.canonical_eyes)
        aligned_color = cv2.warpAffine(image, M, self.canonical_size, flags=cv2.INTER_LINEAR)
        aligned_gray = cv2.warpAffine(gray, M, self.canonical_size, flags=cv2.INTER_LINEAR)
        
        return aligned_color, aligned_gray, M

    def get_skin_ratio(self, color_patch):
        """Calculate the ratio of skin-colored pixels in a patch."""
        ycrcb = cv2.cvtColor(color_patch, cv2.COLOR_BGR2YCrCb)
        # Define skin color bounds in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        return np.sum(mask > 0) / (mask.size + 1e-6)

    def partition_and_assess_occlusion(self, aligned_color, aligned_gray):
        """Partition into patches and detect occlusion using skin color & texture."""
        face_patches = {}
        visibility = {}
        
        for name, (x, y, w, h) in self.patches.items():
            patch_gray = aligned_gray[y:y+h, x:x+w]
            patch_color = aligned_color[y:y+h, x:x+w]
            face_patches[name] = patch_gray
            
            # 1. Edge density, Variance, & Entropy check (for featureless or artificial occluders)
            laplacian = cv2.Laplacian(patch_gray, cv2.CV_64F)
            edge_density = laplacian.var()
            intensity_std = np.std(patch_gray)
            
            # Shannon entropy measures the information content/complexity.
            # Very low entropy = flat occluder; Very high entropy = complex occluder (e.g. textured mask)
            entropy_val = shannon_entropy(patch_gray)
            
            # 2. Skin ratio check (for textured occluders like masks, hands, scarves)
            skin_ratio = self.get_skin_ratio(patch_color)
            
            is_visible = True
            
            # Thresholds: Added entropy condition
            if edge_density < 30 or intensity_std < 10 or entropy_val < 3.5:
                is_visible = False
                
            # Eyes and mouth naturally have less "skin" color, so apply skin check strictly to cheeks and nose
            if name in ['left_cheek', 'right_cheek', 'nose']:
                if skin_ratio < 0.25: # Yüzde 25'ten az ten rengi varsa muhtemelen maske/el/eşarp var
                    is_visible = False
            elif name in ['mouth']:
                if skin_ratio < 0.15: # Ağız bölgesi için daha esnek ten rengi kontrolü
                    is_visible = False

            visibility[name] = is_visible
            
        # Attempt symmetry recovery for severe occlusion
        for name in list(self.patches.keys()):
            if not visibility[name] and name in self.symmetry_map:
                sym_name = self.symmetry_map[name]
                if visibility[sym_name]:
                    # Mirror the symmetric patch
                    face_patches[name] = cv2.flip(face_patches[sym_name], 1)
                    visibility[name] = True # Recovered
                    
        return face_patches, visibility

    def extract_features(self, patches, visibility):
        """Extract LBP, HOG, and Gabor features for visible patches."""
        features = {}
        for name, patch in patches.items():
            if visibility[name]:
                # Resize for consistent feature size just in case
                patch = cv2.resize(patch, (64, 64)) # Use power of 2 for better HOG
                
                # LBP (Primary)
                radius = 2 # Artırıldı
                n_points = 8 * radius
                lbp = local_binary_pattern(patch, n_points, radius, method='uniform')
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
                lbp_feat = hist.astype("float")
                lbp_feat /= (lbp_feat.sum() + 1e-6)
                
                # HOG (Secondary)
                # Daha iyi ayarlar: 64x64 patch, 8x8 cell, 2x2 blocks per cell
                hog_feat = hog(patch, orientations=8, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False, feature_vector=True)
                
                # Normalize HOG manually just in case
                hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-6)
                
                # Gabor Wavelet Filters (Tertiary)
                gabor_feat = []
                for kern in self.gabor_filters:
                    fimg = cv2.filter2D(patch, cv2.CV_8UC3, kern)
                    gabor_feat.extend([fimg.mean(), fimg.var()])
                gabor_feat = np.array(gabor_feat)
                gabor_feat = gabor_feat / (np.linalg.norm(gabor_feat) + 1e-6)
                
                # Concatenate features (Ağırlıklandırılmış birleştirme)
                features[name] = np.concatenate([lbp_feat * 0.35, hog_feat * 0.45, gabor_feat * 0.20])
            else:
                features[name] = None
                
        return features

    def chi_square_distance(self, histA, histB):
        """Compute the Chi-Square distance between two histograms."""
        eps = 1e-10
        # Compute chi-squared distance
        d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
        return d

    def match(self, features1, vis1, features2, vis2):
        """Compare features using weighted Chi-Square similarity."""
        total_score = 0
        valid_patches = 0
        
        for name in self.patches.keys():
            if vis1[name] and vis2[name]:
                f1 = features1[name]
                f2 = features2[name]
                
                # Calculate Chi-Square distance
                dist = self.chi_square_distance(f1, f2)
                
                # Convert distance to similarity (empirical scaling)
                sim = np.exp(-dist * 2.5) # Scale factor to match 0-1 threshold behavior
                
                # Information-Theoretic heuristic weights
                # Eyes are most unique, nose is stable, cheeks are prone to varied lighting
                weight = 1.8 if name in ['left_eye', 'right_eye'] else (1.4 if name == 'nose' else 1.0)
                
                total_score += sim * weight
                valid_patches += weight
                
        if valid_patches == 0:
            return 0.0 # No common visible patches
            
        return total_score / valid_patches

    def verify(self, img_path1, img_path2, threshold=0.75):
        """Full pipeline to verify two images."""
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images.")
            
        aligned_color1, aligned_gray1, _ = self.detect_and_align(img1)
        aligned_color2, aligned_gray2, _ = self.detect_and_align(img2)
        
        if aligned_color1 is None or aligned_color2 is None:
            return "UNKNOWN (Face not detected)", 0.0, None, None
            
        patches1, vis1 = self.partition_and_assess_occlusion(aligned_color1, aligned_gray1)
        patches2, vis2 = self.partition_and_assess_occlusion(aligned_color2, aligned_gray2)
        
        feat1 = self.extract_features(patches1, vis1)
        feat2 = self.extract_features(patches2, vis2)
        
        score = self.match(feat1, vis1, feat2, vis2)
        
        result = "SAME" if score > threshold else "DIFFERENT"
        
        # Debug images showing alignment and visibility
        debug1 = self.draw_debug(aligned_color1, vis1)
        debug2 = self.draw_debug(aligned_color2, vis2)
        
        return result, score, debug1, debug2
        
    def draw_debug(self, aligned_color, visibility):
        """Draw bounding boxes around patches with color coding for visibility."""
        debug_img = aligned_color.copy()
        for name, (x, y, w, h) in self.patches.items():
            color = (0, 255, 0) if visibility[name] else (0, 0, 255) # Green=Visible, Red=Occluded
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, name.split('_')[0], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return debug_img

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Kullanım: python face_verification.py <resim_yolu_1> <resim_yolu_2>")
        sys.exit(1)
        
    img_path1 = sys.argv[1]
    img_path2 = sys.argv[2]
    
    print(f"Sistem başlatılıyor...")
    verifier = ClassicalFaceVerifier()
    
    print(f"Karşılaştırılıyor: {img_path1} vs {img_path2}")
    try:
        result, score, debug1, debug2 = verifier.verify(img_path1, img_path2)
        
        print("\n" + "="*30)
        print(f"SONUÇ: {result}")
        print(f"GÜVEN PUANI: {score:.4f}")
        print("="*30)
        
        if debug1 is not None:
            cv2.imwrite("debug_1.jpg", debug1)
            cv2.imwrite("debug_2.jpg", debug2)
            print("Hata ayıklama görüntüleri 'debug_1.jpg' ve 'debug_2.jpg' olarak kaydedildi.")
            
    except Exception as e:
        print(f"Hata oluştu: {e}")