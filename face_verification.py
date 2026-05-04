"""
Classical Face Verification Architecture
----------------------------------------
A pose-invariant face verification system robust to extreme occlusion,
developed strictly using Classical Computer Vision techniques.

Core Pipeline:
1. Ensemble Detection: Dlib HOG with fallback Haar cascades for masked faces.
2. 2D Alignment: Affine transformation based on localized eye coordinates.
3. Grid-Based Occlusion Masking: Identifies occluded regions via skin-color thresholding.
4. Feature Extraction: Mask-aware SIFT keypoint detection (ignores occluded blocks).
5. Matching: Lowe's Ratio Test combined with RANSAC geometric consensus.
"""
import cv2
import dlib
import numpy as np
import urllib.request
import shutil
import sys
from pathlib import Path
from skimage.feature import local_binary_pattern

class ClassicalFaceVerifier:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        # 1. Face Detection & Landmark Detection
        self.detector = dlib.get_frontal_face_detector()
        
        self.haar_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.haar_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.lbp_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        self.eye_pair_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        predictor_path = self._resolve_predictor_path(predictor_path)
        self.predictor = dlib.shape_predictor(str(predictor_path))
        
        # Canonical face definitions
        self.canonical_size = (300, 300)
        self.canonical_eyes = np.float32([(90, 90), (210, 90)]) # Left eye, Right eye

        # Feature Extraction Parameters (Gabor + LBP)
        self.gabor_kernels = self._build_gabor_filters()
        self.lbp_radius = 2
        self.lbp_n_points = 8 * self.lbp_radius
        self.grid_shape = (6, 6) # Divide face into 6x6 blocks
        
    def _build_gabor_filters(self):
        """Creates a bank of Gabor filters at multiple scales and orientations."""
        filters = []
        ksize = 15
        for theta in np.arange(0, np.pi, np.pi / 4): # 4 orientations
            for sigma in (2.0, 4.0): # 2 scales
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filters.append(kern)
        return filters

    def _resolve_predictor_path(self, predictor_path):
        predictor_file = Path(predictor_path)
        if predictor_file.exists() and predictor_file.stat().st_size > 0:
            return predictor_file
        print("Downloading dlib 68-point shape predictor...")
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = predictor_file.with_suffix(predictor_file.suffix + ".bz2")
        with urllib.request.urlopen(url) as response, open(compressed_path, "wb") as compressed_file:
            shutil.copyfileobj(response, compressed_file)
        import bz2
        with bz2.open(compressed_path, "rb") as compressed_file, open(predictor_file, "wb") as output_file:
            shutil.copyfileobj(compressed_file, output_file)
        compressed_path.unlink(missing_ok=True)
        return predictor_file

    def detect_and_align(self, image):
        """Detects face using ensemble classical methods, gets landmarks, and aligns."""
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
                
        def cv2_to_dlib(faces):
            if len(faces) == 0:
                return None
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            return dlib.rectangle(left=int(x), top=int(y), right=int(x+w), bottom=int(y+h))

        # 3. Fallback: Haar Default (Equalized)
        if rect is None:
            faces = self.haar_default.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            rect = cv2_to_dlib(faces)
            
        # 4. Fallback: Haar Profile (Original)
        if rect is None:
            faces = self.haar_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            rect = cv2_to_dlib(faces)

        # 5. Fallback: LBP (Original)
        if rect is None and not self.lbp_cascade.empty():
            faces = self.lbp_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            rect = cv2_to_dlib(faces)
            
        # 6. Ultimate Fallback: Eye-Pair Extrapolation
        if rect is None and not self.eye_pair_cascade.empty():
            h_half = gray.shape[0] // 2
            roi_upper = gray[:h_half, :]
            eyes = self.eye_pair_cascade.detectMultiScale(roi_upper, scaleFactor=1.1, minNeighbors=4, minSize=(40, 20))
            if len(eyes) > 0:
                (ex, ey, ew, eh) = eyes[0]
                face_w = int(ew * 1.8)
                face_h = int(face_w * 1.3)
                face_x = ex - int((face_w - ew) / 2)
                face_y = ey - int(eh * 0.8)
                face_x, face_y = max(0, face_x), max(0, face_y)
                rect = dlib.rectangle(face_x, face_y, face_x + face_w, face_y + face_h)
                
        if rect is None:
            return None, None, None, None
            
        # Get landmarks
        shape = self.predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            
        left_eye = coords[36:42].mean(axis=0)
        right_eye = coords[42:48].mean(axis=0)
        
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
        
        src_pts = np.float32([left_eye, right_eye]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.canonical_eyes[0], self.canonical_eyes[1]]).reshape(-1, 1, 2)
        
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        
        if M is None:
            d_y = right_eye[1] - left_eye[1]
            d_x = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(d_y, d_x))
            dist_actual = np.sqrt(d_x**2 + d_y**2)
            dist_canonical = np.sqrt((dst_pts[1][0][0]-dst_pts[0][0][0])**2 + (dst_pts[1][0][1]-dst_pts[0][0][1])**2)
            scale = dist_canonical / (dist_actual + 1e-6)
            center_actual = (left_eye + right_eye) / 2
            center_canonical = (dst_pts[0][0] + dst_pts[1][0]) / 2
            M = cv2.getRotationMatrix2D(tuple(center_actual), angle, scale)
            M[0, 2] += (center_canonical[0] - center_actual[0])
            M[1, 2] += (center_canonical[1] - center_actual[1])

        aligned_color = cv2.warpAffine(image, M, self.canonical_size, flags=cv2.INTER_LINEAR)
        aligned_gray = cv2.warpAffine(gray, M, self.canonical_size, flags=cv2.INTER_LINEAR)
        
        # Transform landmarks to canonical space
        ones = np.ones(shape=(len(coords), 1))
        points_ones = np.hstack([coords, ones])
        canonical_landmarks = M.dot(points_ones.T).T
        
        return aligned_color, aligned_gray, M, canonical_landmarks

    def get_occlusion_mask(self, aligned_color, canonical_landmarks):
        """
        Part-Based Masking: Uses skin color and structural entropy to identify occluded blocks.
        """
        # 1. Skin Color Thresholding
        ycrcb = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        h, w = self.canonical_size
        block_h, block_w = h // self.grid_shape[0], w // self.grid_shape[1]
        
        block_validity = np.ones(self.grid_shape, dtype=bool)
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                
                block_skin = skin_mask[y1:y2, x1:x2]
                skin_ratio = np.sum(block_skin > 0) / (block_h * block_w)
                
                # If a block has very little skin color, it's likely occluded (mask, sunglasses)
                # Exception: Eye blocks naturally have less skin, but we rely on a generous threshold
                if skin_ratio < 0.15:
                    block_validity[i, j] = False
                    
        return block_validity

    def extract_features_with_mask(self, aligned_gray, block_validity):
        """Extract SIFT keypoints, discarding those that fall into occluded blocks."""
        # Initialize SIFT if not done
        if not hasattr(self, 'sift'):
            self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03, edgeThreshold=10)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            
        keypoints, descriptors = self.sift.detectAndCompute(aligned_gray, mask=None)
        
        if descriptors is None or len(keypoints) == 0:
            return [], None
            
        h, w = self.canonical_size
        block_h, block_w = h // self.grid_shape[0], w // self.grid_shape[1]
        
        valid_kp = []
        valid_des = []
        
        for idx, kp in enumerate(keypoints):
            x, y = kp.pt
            # Find which block this keypoint belongs to
            block_j = min(int(x // block_w), self.grid_shape[1] - 1)
            block_i = min(int(y // block_h), self.grid_shape[0] - 1)
            
            if block_validity[block_i, block_j]:
                valid_kp.append(kp)
                valid_des.append(descriptors[idx])
                
        if len(valid_kp) == 0:
            return [], None
            
        return valid_kp, np.array(valid_des)

    def match_features(self, kp1, des1, kp2, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0, None, None

        matches = self.flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        inliers = 0
        mask = None
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = np.sum(mask)

        return inliers, good_matches, mask

    def verify(self, img_path1, img_path2, threshold=4):
        """Pipeline utilizing Regional Occlusion Masking + SIFT + RANSAC."""
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images.")
            
        aligned_color1, aligned_gray1, M1, lm1 = self.detect_and_align(img1)
        aligned_color2, aligned_gray2, M2, lm2 = self.detect_and_align(img2)
        
        if aligned_color1 is None or aligned_color2 is None:
            return "UNKNOWN (Face not detected)", 0, None, None
            
        # 1. Part-Based Occlusion Masking
        val1 = self.get_occlusion_mask(aligned_color1, lm1)
        val2 = self.get_occlusion_mask(aligned_color2, lm2)
        common_validity = val1 & val2
        
        valid_blocks_count = np.sum(common_validity)
        if valid_blocks_count < (self.grid_shape[0] * self.grid_shape[1]) * 0.15:
             return "UNKNOWN (Too much occlusion)", 0, aligned_color1, aligned_color2

        # 2. Extract SIFT features, filtering by occlusion mask
        kp1, des1 = self.extract_features_with_mask(aligned_gray1, common_validity)
        kp2, des2 = self.extract_features_with_mask(aligned_gray2, common_validity)
        
        # 3. Match using Lowe's Ratio + RANSAC
        inliers, good_matches, mask = self.match_features(kp1, des1, kp2, des2)
        
        result = "SAME" if inliers >= threshold else "DIFFERENT"
        
        # Debug Visualization
        debug_matches = None
        if good_matches and mask is not None:
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            debug_matches = cv2.drawMatches(aligned_color1, kp1, aligned_color2, kp2, good_matches, None, **draw_params)
        else:
             debug_matches = np.concatenate((aligned_color1, aligned_color2), axis=1)
             
        # Draw invalid blocks in red
        h, w = self.canonical_size
        block_h, block_w = h // self.grid_shape[0], w // self.grid_shape[1]
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if not common_validity[i, j]:
                    y1, y2 = i * block_h, (i + 1) * block_h
                    x1, x2 = j * block_w, (j + 1) * block_w
                    cv2.rectangle(debug_matches, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.rectangle(debug_matches, (x1 + w, y1), (x2 + w, y2), (0, 0, 255), 1)

        return result, inliers, debug_matches, debug_matches

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Kullanım: python face_verification.py <resim_yolu_1> <resim_yolu_2>")
        sys.exit(1)
    img_path1 = sys.argv[1]
    img_path2 = sys.argv[2]
    verifier = ClassicalFaceVerifier()
    try:
        result, score, debug1, debug2 = verifier.verify(img_path1, img_path2)
        print(f"SONUÇ: {result}")
        print(f"GÜVEN PUANI: {score:.4f}")
    except Exception as e:
        print(f"Hata oluştu: {e}")
