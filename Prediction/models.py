import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


def getReports(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f'Confusion Matrix:\n{cm}\n\nClassification Report:\n{cr}\n')


def loadModel(path):
    model = joblib.load(path)
    return model


def deployModel(model, path):
    joblib.dump(model, path)


def gbcModel(modelPath, x_train, x_test, y_train, y_test):
    print("\t\t\t*** GBC MODEL ***\n")

    scaler = StandardScaler()

    print(f'X_test:\n{x_test}\n\nx_train:\n{x_train}')
    x_train = scaler.fit_transform(x_train)

    print(f'X_test:\n{x_test}\n\nx_train:\n{x_train}')

    x_test = scaler.transform(x_test)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    # Create the GradientBoostingClassifier
    gbm = GradientBoostingClassifier()

    # Perform Grid Search
    grid_search = GridSearchCV(gbm, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Access the results
    results = grid_search.cv_results_

    # Display the progress
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f'Mean Score: {mean_score}\nParameters: {params}\n')

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'\nBest Score: {best_score}\n')

    # Train the model with the best parameters
    best_gbm = GradientBoostingClassifier(**best_params)
    best_gbm.fit(x_train, y_train)

    # Evaluate the model
    test_score = best_gbm.score(x_test, y_test)

    print("Best Parameters: ", best_params)
    print("Best Score: ", best_score)
    print("Test Score with Best Parameters: ", test_score)

    y_pred = best_gbm.predict(x_test)

    getReports(y_test, y_pred)

    probs = best_gbm.predict_proba(x_test)
    probs = probs[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f'\nAUC: {auc * 100}\n')

    deployModel(best_gbm, modelPath)
