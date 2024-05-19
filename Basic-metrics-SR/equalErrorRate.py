import csv

SCORES = 'scores.csv'

def ComputeEER():
    """Compute the Equal Error Rate from the data in scores.csv
    
    Returns:
        a floating point number for the equal error rate (between 0 and 1)
    """
    labels = []
    scores = []
    with open(SCORES) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            labels.append(int(row[0]))
            scores.append(float(row[1]))

    # Sort scores in descending order
    sorted_scores = sorted(scores, reverse=True)

    eer = 1.0  # Initialize EER to 100%
    eer_threshold = None
    
    # Sweep through thresholds
    for threshold in sorted_scores:
        far = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 0) / sum(1 for score, label in zip(scores, labels) if label == 0)
        frr = sum(1 for score, label in zip(scores, labels) if score < threshold and label == 1) / sum(1 for score, label in zip(scores, labels) if label == 1)

        # far = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 0) / sum(1 for label in labels if label == 0)
        # frr = sum(1 for score, label in zip(scores, labels) if score < threshold and label == 1) / sum(1 for label in labels if label == 1)
        delta = abs(far - frr)
        
        # Update EER if the delta between FAR and FRR is minimal
        if delta < eer:
            eer = delta
            eer_threshold = threshold

    return eer, eer_threshold

# Example usage:
eer, eer_threshold = ComputeEER()
print("Equal Error Rate (EER):", eer)
print("EER Threshold:", eer_threshold)
