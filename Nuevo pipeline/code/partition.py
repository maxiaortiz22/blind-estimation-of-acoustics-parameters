import pandas as pd


def part_data(data, n_folds, stratify_column, independent_column, random_state=None):
    data['fold'] = -1
    labels = data[stratify_column].unique()
    amount_label_by_fold = pd.DataFrame([[0] * len(labels) for _ in range(n_folds)], columns=list(labels),
                                        index=list(range(n_folds)))

    fold_for_independent = {}
    data = data.sample(frac=1.0, random_state=random_state)
    for i, audio in data.iterrows():
        if fold_for_independent.get(audio[independent_column]):
            fold = fold_for_independent[audio[independent_column]]
        else:
            fold = amount_label_by_fold[[audio[stratify_column]]].idxmin().values[0]
            fold_for_independent[audio[independent_column]] = fold
        data.loc[i, 'fold'] = fold
        amount_label_by_fold.loc[fold, audio[stratify_column]] += 1
    return data
