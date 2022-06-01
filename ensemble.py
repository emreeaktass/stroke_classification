import numpy as np
from sklearn.model_selection import train_test_split
import Dataset as D
import DataLoader as DL
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import os
import warnings
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# X = np.arange(20)
# y = np.ones(20)
# # [ 7  8  3  1 19 11  9 17 13 18  2  4 14 10 12  5]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
# print(X_train)


X_train, y_train = D.get_splitted_datas('train')
X_val, y_val = D.get_splitted_datas('validation')
X_test, y_test = D.get_splitted_datas('test')
train_generator = DL.get_generator(X_train, y_train, 1)
val_generator = DL.get_generator(X_val, y_val, 1)
test_generator = DL.get_generator(X_test, y_test, 1)

inme_yok_diger = tf.keras.models.load_model('record21/', compile=False)
inme_var_diger = tf.keras.models.load_model('record22/', compile=False)
iskemi_diger = tf.keras.models.load_model('record23/', compile=False)
kanama_diger = tf.keras.models.load_model('record24/', compile=False)
y_pred_inme_yok_diger = []
y_pred_inme_var_diger = []
y_pred_iskemi_diger = []
y_pred_kanama_diger = []
# y = []
# X_tra = []
# counter = 0
# for i in train_generator:
#     a = []
#     print(counter)
#     X = i[0]
#
#     a.append(1 - inme_yok_diger.predict(X))
#     a.append(inme_var_diger.predict(X))
#     a.append(iskemi_diger.predict(X))
#     a.append(kanama_diger.predict(X))
#     a = np.array(a).reshape(1, 4)
#     X_tra.append(a)
#     y.append(i[1][0][0])
#     counter = counter + 1
#
#     # if counter == 5000:
#     #     break
#
# X_test = []
# y_test = []
# counter = 0
# for i in test_generator:
#     a = []
#     X = i[0]
#     a.append(1 - inme_yok_diger.predict(X))
#     a.append(inme_var_diger.predict(X))
#     a.append(iskemi_diger.predict(X))
#     a.append(kanama_diger.predict(X))
#     a = np.array(a).reshape(1, 4)
#     X_test.append(a)
#     y_test.append(i[1][0][0])
#     counter = counter + 1

# X_tra = np.array(X_tra)
# X_tra = X_tra.reshape(len(X_tra), 4)

# X_test = np.array(X_test)
# X_test = X_test.reshape(len(X_test), 4)

# print(X_tra)
# print(y)
# print(X_test)
# print(y_test)

# np.save('X_tra', X_tra)
# np.save('y', y)
# np.save('X_test', X_test)
# np.save('y_test', y_test)

X_tra = np.load('X_tra.npy')
y = np.load('y.npy')


X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')



# clf = KNeighborsClassifier()
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'leaf_size': [5, 10, 20, 30, 50, 100, 200, 1000],
#     'p': [1, 2, 3]
# }
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=10,
#     n_jobs=-1,
#     scoring='neg_mean_squared_error',
#     verbose=0
# )
# gs.fit(X_tra, y)
# print('KNN : ', gs.best_estimator_)
#
# clf = LogisticRegression()
# param_grid = {
#     'C': [0.01, 0.1, 1, 2, 5, 10, 50, 100],
#     'fit_intercept': [True, False],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     'max_iter': [50, 100, 200, 500, 1000, 2000]
# }
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=10,
#     n_jobs=-1,
#     scoring='neg_mean_squared_error',
#     verbose=0
# )
# gs.fit(X_tra, y)
# print('Logistic Classifier : ', gs.best_estimator_)
#
# clf = SVC()
# param_grid = {
#     'C': [0.1, 1, 2, 5],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'degree': [1, 2, 3],
#     'decision_function_shape': ['ovo', 'ovr']
# }
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=10,
#     n_jobs=-1,
#     scoring='neg_mean_squared_error',
#     verbose=0
# )
# gs.fit(X_tra, y)
# print('SVC : ', gs.best_estimator_)
#
# clf = RandomForestClassifier()
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [5, 10, 50, None],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=10,
#     n_jobs=-1,
#     scoring='neg_mean_squared_error',
#     verbose=0
# )
# gs.fit(X_tra, y)
# print('Random Forest : ', gs.best_estimator_)
#

# X_tra = X_tra[:, [2,3]]

# X_test = X_test[:, [2,3]]
counter = 0
em = []
count = 0
for j,i in enumerate(X_tra):

    # if (i[2] == 1 or i[3] == 1):
    #     if i[1] == 1:
    #         print('emre')
    # if i[2] == 1 and i[3] == 1:
    #     print('aktas')
    if i[0] > 0.5:
        a = 0
    else:
        a = 1

    if i[1] > 0.5:
        b = 1
    else:
        b = 0

    if i[2] > 0.5:
        c = 1
    else:
        c = 0

    if i[3] > 0.5:
        d = 1
    else:
        d = 0
    # if y[j] == b:
    #     print(a,b,c,d, y[j])
    #     counter = counter + 1

    # if a == 0 and b == 0 and c == 0 and d == 0:
    #     em.append(0)
    # elif b == 1 and (c == 1 or d == 1):
    #     count += 1
    #     em.append(1)
    # else:
    #     em.append(0)
    res = 0
    # y sağlıklı ise 1 sağlıksız ise 0, a
    if a == 0:
        if b == 0:
            if c or d:
                em.append(0)
                res = 0
            else:
                em.append(1)
                res = 1
        else:
            em.append(0)
            res = 0
    else:
        if b == 1:
            if c or d:
                em.append(0)
                res = 0
            else:
                em.append(1)
                res = 1
        else:
            if c or d:
                em.append(0)
                res = 0
            else:
                em.append(1)
                res = 1

        if res != y_test[j]:
            counter+=1
            print(a, b, c, d, res, y_test[j])

    # a 1 iken 7 sıkıntılı, 0 iken 2 sıkıntılı
    # a 1 iken 334 sıkıntılı, 0 iken 125 sıkıntılı
    # 486*47

    # if (i[1] >= 0.7 and i[2] >= 0.7) or (i[1] >= 0.7 and i[3] >= 0.7) or (i[2] >= 0.7 and i[3] >= 0.7):
    # if (b == 1 and c == 1) or (b == 1 and d == 1) or (c == 1 and d == 1) or (c == 1 or d == 1):
    #         em.append(0)
    #         if y_test[j] == 1:
    #             print('sağlıklı ama inme predict : ', a, b, c, d)
    #         # print(a,b,c,d)
    #     # if a == 0:
    #     #     em.append(0)
    #     # else:
    #     #     em.append(1)
    # else:
    #     # 0110 0101 0011
    #     if (c == 0 and d == 0 and (a != 0 or b != 0)) or (c == 0 and d == 0 and a == 0 and b == 0):
    #         em.append(0)
    #
    #     elif a == 1:
    #         em.append(1)
    #         if y_test[j] == 0:
    #             print('inme var ama sağlıklı predict : ', a, b, c, d, y_test[j])
    #     else:
    #         em.append(0)
    #         if y_test[j] == 1:
    #             print('sağlıklı ama sağlıksız predict else : ', a, b, c, d, y_test[j])
        # if (a == 0 and d == 1) or (a == 0 and c == 1):
        #     em.append(0)
        #     # print(a, b, c, d)
        # else:
        #     print(a, b, c, d)
        #     em.append(1)
        # if y_test[j] == 0:
        #     print('inme var ama sağlıklı predict : ', a, b, c, d)
#     if ((b == 1 and c == 1) or (b == 1 and d == 1)) and y[j] == 0:
#         counter = counter + 1
#         print(a, b, c, d, y[j])
# print(counter)
# print(confusion_matrix(y_test, em))
# print(sum(em))
print(sum(y))
print(counter)
print(confusion_matrix(y_test, em))

# kanama varken sağlıklı olan sayısı 3
# iskemi varken sağlıklı olan sayısı 51
# inme varken sağlıklı olan sayısı 77
# iskemi ve kanama varken sağlıklı olan sayısı 0
# kanama ve inme varken sağlıklı sayısı 1
# iskemi ve inme varken sağlıklı sayısı 4
# iskemi ve kanama yokken sağlıksız olan sayısı 711
# inme varken ve yokken sağlıklı olan sayısı 41
# inme varken ve yokken sağlıksız olan sayısı 155




# clf = AdaBoostClassifier()
# param_grid = {
#     'n_estimators': [10, 30, 50, 100, 200],
#     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#     'algorithm': ['SAMME', 'SAMME.R']
# }
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=10,
#     n_jobs=-1,
#     scoring='accuracy',
#     return_train_score=True,
#     verbose=1
# )
# gs.fit(X_tra, y)
# print('Ada Boost : ', gs.best_estimator_)
# #
# clf = GradientBoostingClassifier()
# param_grid = {
#     'loss': ['deviance', 'exponential'],
#     'n_estimators': [10, 50, 100, 200],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'max_depth': [2, 3, 5, 10, 50],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# gs = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     cv=10,
#     n_jobs=-1,
#     scoring='accuracy',
#     verbose=1
# )
# gs.fit(X_tra, y)
# print('Gradient Boosting : ', gs.best_estimator_)

# lr = LogisticRegression(C=50, max_iter=50, solver='newton-cg')
# lr.fit(X_tra, y)
# y_pred = lr.predict(X_test)
# print(confusion_matrix(y_pred, y_test))


# svc = SVC(C=5, decision_function_shape='ovo', degree=1, kernel='linear')
# svc.fit(X_tra, y)
# y_pred = svc.predict(X_test)
# print(confusion_matrix(y_pred, y_test))

# random = RandomForestClassifier(max_depth=5, max_features='log2', n_estimators=200)
# random.fit(X_tra, y)
# y_pred = random.predict(X_test)
# print(confusion_matrix(y_pred, y_test))
#
# knn = KNeighborsClassifier(leaf_size=5, p=3)
# knn.fit(X_tra, y)
# y_pred = knn.predict(X_test)
# print(confusion_matrix(y_pred, y_test))
#
# ada = AdaBoostClassifier(learning_rate=0.1, n_estimators=30)
# # # lr = AdaBoostClassifier(learning_rate=0.3)
# ada.fit(X_tra, y)
# y_pred = ada.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
#
# boost = GradientBoostingClassifier(loss='exponential', max_depth=2, max_features='sqrt',n_estimators=50)
# boost.fit(X_tra, y)
# y_pred = boost.predict(X_test)
# print(confusion_matrix(y_test, y_pred))


# X = []
# for i,j in zip(X_test, y_test):
#     if i[0] < 0.5:
#         a = 0
#     else:
#         a = 1
#     X.append(a)


from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X_tra)

# y_pred = kmeans.predict(X_test)
# a = 1 - y_pred
# print(confusion_matrix(a, y_test))
# print([X_test[0]])
# a = np.array(X_test[0]).astype('float64')
# a = np.array(X_test[0], dtype=np.double)
# print(a.dtype)
# s = kmeans.predict([a])
# pkl_filename = "inme_var_yok2.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(clf, file)


# Logistic Classifier :  LogisticRegression(C=50, fit_intercept=False, max_iter=50, solver='newton-cg')
# SVC :  SVC(C=5, decision_function_shape='ovo', degree=1, gamma='auto', kernel='linear')
# Random Forest :  RandomForestClassifier(criterion='entropy', max_depth=10, max_features='log2',
#                        n_estimators=50)
# KNN :  KNeighborsClassifier(leaf_size=5, n_neighbors=15, p=3)
# Ada Boost :  AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
# Gradient Boosting :  GradientBoostingClassifier(max_features='log2', n_estimators=10)


# KNN :  KNN :  KNeighborsClassifier(leaf_size=5, p=3)
# Logistic Classifier :  LogisticRegression(C=50, max_iter=50, solver='newton-cg')
# SVC :  SVC(C=5, decision_function_shape='ovo', degree=1, kernel='linear')
# Random Forest :  RandomForestClassifier(max_depth=5, max_features='log2', n_estimators=200)
# Ada Boost :  AdaBoostClassifier(learning_rate=0.1)
# last = []
# for z, i in enumerate(X_test):
#     a = lr.predict([i])
#     b = ada.predict([i])
#     c = em[z]
#     d = random.predict([i])
#
#     res = (a + b + c + d) / 4
#
#     if res >= 0.5:
#         last.append(1)
#     else:
#         last.append(0)
#
#     # if res != y_test[z]:
#     #     print(y_test[z], a, b, c)
# print(confusion_matrix(last, y_test))
