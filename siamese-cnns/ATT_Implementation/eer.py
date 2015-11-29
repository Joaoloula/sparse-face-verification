import numpy as np

max_threshold = 200  # Maximum threshold for energy
step = 1             # Step for FAR and FRR computation


def far_and_frr(scores, labels, threshold):
    # Given a scores array, a labels array and a threshold, computes the false
    # acceptance and false rejection rates
    sample_size = len(scores)
    assert sample_size == len(labels)

    [positives, negatives, false_positives, false_negatives] = [0., 0., 0., 0.]

    for i in range(sample_size):
        if labels[i] == 0:
            positives += 1
            if scores[i] >= threshold:
                false_negatives += 1
        else:
            negatives += 1
            if scores[i] < threshold:
                false_positives += 1
    if negatives != 0:
        far = false_positives/negatives
    else:
        far = 0
    if positives != 0:
        frr = false_negatives/positives
    else:
        frr = 0

    return [far, frr]


def eer(scores, labels):
    # Calculates the EER from an array of scores and the true labels
    [mean, difference] = [0.5, 1.0]
    for i in np.arange(0, max_threshold, step):
        [far, frr] = far_and_frr(scores, labels, i)
        if abs(far-frr) < difference:
            difference = abs(far-frr)
            mean = (far+frr)/2
    return mean
