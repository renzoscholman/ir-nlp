import functools
import operator
import random


# Custom fold generator preventing information leakage by keeping claim ids in their own folds
def cv_fold_generator(data, n_folds=10):
    claim_dict = {}

    # Group data by claim ID in a dict
    for i in range(0, len(data)):
        claim_id = data[i]
        index = i
        if not claim_id in claim_dict:
            claim_dict[claim_id] = [index]
        else:
            claim_dict[claim_id].append(index)

    # Convert dict to list for custom order
    claim_list = list(claim_dict.items())

    # Shuffle list for randomization, use seed for reproducibility
    random.seed(1)
    random.shuffle(claim_list)

    # Placeholder for folds
    folds = []

    # Counter for iterating through randomized claim list
    counter = 0

    # Desired size of each fold
    fold_size = len(data) / n_folds

    # Separate list into folds using greedy approach, it stops filling a fold after it is equal or passed the fold size
    for i in range(n_folds):
        fold = []
        # First n_folds - 1 folds, fill fold with at least fold_size data rows
        if i < n_folds - 1:
            while len(fold) < fold_size:
                fold = fold + (claim_list[counter][1])
                counter = counter + 1
        # Last fold, fills it with remaining data rows
        elif i == n_folds - 1:
            while counter < len(claim_list):
                fold = fold + (claim_list[counter][1])
                counter = counter + 1
        # This should never execute
        else:
            raise Exception("Cross validation fold generation error")
        folds.append(fold)

    # Folds containing train and test indices
    cv = []

    # Construct the train and test indices for each fold
    for i in range(0, len(folds)):
        # Fold i as test fold
        test_fold = folds[i]
        # Remaining folds as train folds
        train_folds = functools.reduce(operator.iconcat, folds[:i] + folds[i + 1:], [])  # get all folds except fold i
        # Append folds to cv
        cv.append((train_folds, test_fold))

    return cv
