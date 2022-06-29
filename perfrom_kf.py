from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def perform_kf(XX, yy, my_model, nn, ns):
    kf = KFold(n_splits = ns);
    mae_train = [];
    mae_test = [];
    for train_index, test_index in kf.split(XX):
        XX_train, XX_test = XX[train_index], XX[test_index];
        yy_train, yy_test = yy[train_index], yy[test_index];
        my_model.fit(XX_train, yy_train, epochs = nn);
        yy_train_pred = my_model.predict(XX_train);
        yy_test_pred = my_model.predict(XX_test);
        curr_mae_train = mean_absolute_error(yy_train, yy_train_pred);
        curr_mae_test = mean_absolute_error(yy_test, yy_test_pred);
        mae_train.append(curr_mae_train);
        mae_test.append(curr_mae_test);
    return np.array(mae_train), np.array(mae_test);
