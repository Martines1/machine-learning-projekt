from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Linear Support Vector Machine
def get_linearSVC_model(X_train_vec, y_train, C, penalty, loss):
    linear_svc_model_ = LinearSVC(C=C, penalty=penalty, loss=loss, random_state=42, verbose=True)
    linear_svc_model_.fit(X_train_vec, y_train)
    return linear_svc_model_, "Linear SVC"

def find_linearSVC_model(X_train_vec, y_train):
    linear_svc_param_grid = {
        'C': [0.01,0.05, 0.1, 0.5, 1, 10, 20, 50],
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge']
    }
    linear_svc_grid_search = GridSearchCV(
        estimator=LinearSVC(random_state=42, verbose=True),
        param_grid=linear_svc_param_grid,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )
    linear_svc_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for Linear SVC:", linear_svc_grid_search.best_params_)
    return linear_svc_grid_search.best_estimator_, "Linear SVC"

#Support Vector Machine
def get_SVC_model(X_train_vec, y_train,kernel, C, gamma):
    svc_model_ = SVC(kernel=kernel, C = C, gamma=gamma, random_state=42, verbose=True, shrinking=False)
    svc_model_.fit(X_train_vec, y_train)
    return svc_model_, "SVC"

def find_SVC_model(X_train_vec, y_train):
    svc_param_grid = {
        'C': [0.1, 1, 10, 20, 50],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    svc_grid_search = GridSearchCV(
        estimator=SVC(random_state=42, verbose=True, shrinking=False),
        param_grid=svc_param_grid,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )
    svc_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for SVC:", svc_grid_search.best_params_)
    return svc_grid_search.best_estimator_, "SVC"

#Decision Tree
def get_DT_model(X_train_vec, y_train, max_depth, criterion, min_samples_leaf):
    dt_model_ = DecisionTreeClassifier(random_state=42, max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)
    dt_model_.fit(X_train_vec, y_train)
    return dt_model_, "DT"

def find_DT_model(X_train_vec, y_train):
    dt_param_grid = {'max_depth': [3, 5, 10, 15, 20, 30], 'criterion': ["gini", "entropy"], 'min_samples_leaf' : [1,2,3,5]}
    dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
    dt_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for DT:", dt_grid_search.best_params_)
    return dt_grid_search.best_estimator_, "DT"

# Random Forest
def get_RF_model(X_train_vec, y_train,max_depth, n_estimators, criterion):
    rf_model_ = RandomForestClassifier(random_state=42, n_jobs=-1, criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
    rf_model_.fit(X_train_vec, y_train)
    return rf_model_, "RF"

def find_RF_model(X_train_vec, y_train):
    rf_param_grid = {'max_depth': [3, 5, 10, 15, 20], 'n_estimators': [5, 10, 50, 100], 'criterion': ["gini", "entropy"]}
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
    rf_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for RF:", rf_grid_search.best_params_)
    return rf_grid_search.best_estimator_, "RF"

#Gradient Boosting
def get_GB_model(X_train_vec, y_train,learning_rate, max_depth):
    gb_model_ = GradientBoostingClassifier(random_state=42, learning_rate=learning_rate, max_depth=max_depth)
    gb_model_.fit(X_train_vec, y_train)
    return gb_model_, "GB"

def find_GB_model(X_train_vec, y_train):
    gb_param_grid = {
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    }
    gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3, scoring='accuracy',
                            n_jobs=-1, verbose=3)
    gb_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for DT:", gb_grid_search.best_params_)
    return gb_grid_search.best_estimator_, "GB"

#Logistic Regression
def get_LR_model(X_train_vec, y_train, C, penalty, solver):
    lr_model_ = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=C, penalty=penalty, solver=solver, n_jobs=-1)
    lr_model_.fit(X_train_vec, y_train)
    return lr_model_, "LR"

def find_LR_model(X_train_vec, y_train):
    lr_param_grid = {
        'C': [0.01, 0.1, 0.4, 1, 10, 20, 50],
        'penalty': ['l1', 'l2'],
        'solver': ['lbfgs', 'saga', 'newton-cg', 'sag'],
    }
    lr_grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1),
        param_grid=lr_param_grid,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )
    lr_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for LR:", lr_grid_search.best_params_)
    return lr_grid_search.best_estimator_, "LR"

#Naive Bayes
def get_NB_model(X_train_vec, y_train,alpha):
    nb_model_ = MultinomialNB(alpha=alpha)
    nb_model_.fit(X_train_vec, y_train)
    return nb_model_, "NB"

def find_NB_model(X_train_vec, y_train):
    nb_param_grid = {
        'alpha': [round(x, 3) for x in [0.300 + i * 0.001 for i in range(101)]]
    }
    nb_grid_search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=nb_param_grid,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )
    nb_grid_search.fit(X_train_vec, y_train)
    print("Best parameters for NB:", nb_grid_search.best_params_)
    return nb_grid_search.best_estimator_, "NB"