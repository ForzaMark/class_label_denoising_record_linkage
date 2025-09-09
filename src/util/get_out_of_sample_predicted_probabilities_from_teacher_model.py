from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_out_of_sample_predicted_probabilities_from_teacher_model(data, clf, stratified_training = True, scale_feature_data = False):
    result_data = data.copy()

    kf = KFold(n_splits=10, shuffle=True)

    for _, (train_index, test_index) in enumerate(kf.split(data)):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        x_train = train_data.drop(["label"], axis=1)
        x_test = test_data.drop(["label"], axis=1)

        y_train = train_data["label"]

        assert len([col for col in x_train.columns if 'label' in col]) == 0
        assert len([col for col in x_test.columns if 'label' in col]) == 0

        train_columns = x_train.columns

        if scale_feature_data:
            x_train = StandardScaler().fit_transform(x_train)
            x_test = StandardScaler().fit_transform(x_test)
    
        if stratified_training:        
            sampler = RandomUnderSampler()
            X_res, y_res = sampler.fit_resample(x_train, y_train)
            x_train = pd.DataFrame(X_res, columns=train_columns)
            y_train = y_res

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        for i, ti in enumerate(test_index):
            result_data.loc[ti, "predicted_label"] = y_pred[i]

    return result_data