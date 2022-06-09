def evaluate_performance(estimate, target):
    """Create confusion matrix and calculate performance measures

    Args:
        estimate (list): calculated estimate of values
        target (list):   target values
    
    Output:
        sensitivity, specificity, precision, accuracy (double)
        [true_pos, false_pos, true_neg, false_neg]: values for ROC curve 
    """
    
    length = len(estimate)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    
    for n in range(length):
        if estimate[n] == 0:
            if target[n] == 0:
                true_neg += 1
            else:
                false_neg += 1
        else:
            if target[n] == 0:
                false_pos += 1
            else:
                true_pos += 1
    
    if (true_pos + false_neg) > 0: 
        sensitivity = format(true_pos / (true_pos + false_neg), '.5f')
    else:
        sensitivity = 0
        
    if (true_neg + false_pos) > 0: 
        specificity = format(true_neg / (true_neg + false_pos), '.5f')
    else:
        specificity = 0
        
    if (true_pos + false_pos) > 0: 
        precision   = format(true_pos / (true_pos + false_pos), '.5f')
    else:
        precision = 0
        
    if length > 0: 
        accuracy    = format((true_pos + true_neg) / length, '.5f')
    else:
        accuracy    = 0
        
    return sensitivity, specificity, precision, accuracy, [true_pos, false_pos, true_neg, false_neg]