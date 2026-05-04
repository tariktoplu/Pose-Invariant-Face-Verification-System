"""
Dataset Evaluation Module
-------------------------
Evaluates the ClassicalFaceVerifier pipeline on a given dataset.
Reports accuracy, False Acceptance Rate (FAR), and False Rejection Rate (FRR).
"""
import os
import random
import sys
from pathlib import Path
import time

from face_verification import ClassicalFaceVerifier

def get_image_paths(base_dir):
    identities = {}
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir):
            images = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
            if len(images) > 0:
                identities[person_name] = images
    return identities

def generate_pairs(identities, max_pos=50, max_neg=50):
    positive_pairs = []
    negative_pairs = []
    
    names = list(identities.keys())
    
    # Generate Positive Pairs
    for name, images in identities.items():
        if len(images) >= 2:
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    positive_pairs.append((images[i], images[j], 1)) # 1 means SAME
                    
    random.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:max_pos]
    
    # Generate Negative Pairs
    while len(negative_pairs) < max_neg:
        name1, name2 = random.sample(names, 2)
        img1 = random.choice(identities[name1])
        img2 = random.choice(identities[name2])
        negative_pairs.append((img1, img2, 0)) # 0 means DIFFERENT
        
    return positive_pairs + negative_pairs

def evaluate():
    base_dir = "dataset/lfw_masked/lfw_test"
    print(f"Loading dataset from {base_dir}...")
    identities = get_image_paths(base_dir)
    print(f"Found {len(identities)} identities.")
    
    # Limit pairs for reasonable execution time
    pairs = generate_pairs(identities, max_pos=100, max_neg=100)
    random.shuffle(pairs)
    print(f"Generated {len(pairs)} pairs for evaluation ({sum(1 for p in pairs if p[2] == 1)} positive, {sum(1 for p in pairs if p[2] == 0)} negative).")
    
    print("\nInitializing ClassicalFaceVerifier (this may take a moment)...")
    verifier = ClassicalFaceVerifier()
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    failed_detections = 0
    
    start_time = time.time()
    
    print("\nStarting evaluation...")
    for i, (img1, img2, is_same) in enumerate(pairs):
        try:
            # We don't need debug images for evaluation, just the result
            result, score, _, _ = verifier.verify(img1, img2)
            
            if "UNKNOWN" in result:
                failed_detections += 1
                continue
                
            predicted_same = 1 if result == "SAME" else 0
            
            if is_same == 1 and predicted_same == 1:
                true_positives += 1
            elif is_same == 1 and predicted_same == 0:
                false_negatives += 1
            elif is_same == 0 and predicted_same == 0:
                true_negatives += 1
            elif is_same == 0 and predicted_same == 1:
                false_positives += 1
                
        except Exception as e:
            failed_detections += 1
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(pairs)} pairs...")

    total_valid = true_positives + true_negatives + false_positives + false_negatives
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS (Masked LFW)")
    print("="*40)
    print(f"Total Pairs Evaluated : {len(pairs)}")
    print(f"Successful Detections : {total_valid} ({(total_valid/len(pairs))*100:.1f}%)")
    print(f"Failed Detections     : {failed_detections} (Images where faces couldn't be found)")
    
    if total_valid > 0:
        accuracy = (true_positives + true_negatives) / total_valid
        far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        frr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        print(f"\nAccuracy (Doğruluk)   : %{accuracy*100:.2f}")
        print(f"FAR (Yanlış Kabul)    : %{far*100:.2f}")
        print(f"FRR (Yanlış Red)      : %{frr*100:.2f}")
        print("-" * 40)
        print(f"True Positives (Doğru Eşleşme) : {true_positives}")
        print(f"True Negatives (Doğru Red)     : {true_negatives}")
    else:
        print("Not enough valid detections to calculate accuracy.")
        
    print(f"\nTime taken: {time.time() - start_time:.1f} seconds")
    print("="*40)

if __name__ == "__main__":
    evaluate()
