import pandas as pd, seaborn as sb, matplotlib.pyplot as plt, numpy as np, warnings as w, time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.exceptions import DataConversionWarning

# Measuring machine learning model performance using the Wisconsin Breast Cancer dataset
# Libraries/modules = CUDA, cuDNN, Tensorflow, Keras, Scikit-learn

# CLASSES
class Preprocess:
    def __init__(self, file):
        self.train_X, self.train_Y, self.test_X, self.test_Y, self.feature_names = self.file_management(file)

    def file_management(self, file):

        data = pd.read_csv(file, header=None)
        data = data.drop(data.columns[0], axis=1)

        classifier = pd.get_dummies(data[1]).astype(int)

        d_normalised = self.normalise(data.iloc[:, 1:31])
        d_normalised.insert(loc=0, column=32, value=classifier.iloc[:, 0])
        d_normalised.insert(loc=0, column=33, value=classifier.iloc[:, 1])
        data = d_normalised.T.reset_index(drop=True).T
        data.rename(columns={
            0: 'diagnosis_1',
            1: 'diagnosis_2',
            2: 'radius_mean',
            3: 'texture_mean',
            4: 'perimeter_mean',
            5: 'area_mean',
            6: 'smoothness_mean',
            7: 'compactness_mean',
            8: 'concavity_mean',
            9: 'concave_points_mean',
            10: 'symmetry_mean',
            11: 'fractal_dimension_mean',
            12: 'radius_sqerr',
            13: 'texture_sqerr',
            14: 'perimeter_sqerr',
            15: 'area_sqerr',
            16: 'smoothness_sqerr',
            17: 'compactness_sqerr',
            18: 'concavity_sqerr',
            19: 'concave_points_sqerr',
            20: 'symmetry_sqerr',
            21: 'fractal_dimension_sqerr',
            22: 'radius_worst',
            23: 'texture_worst',
            24: 'perimeter_worst',
            25: 'area_worst',
            26: 'smoothness_worst',
            27: 'compactness_worst',
            28: 'concavity_worst',
            29: 'concave_points_worst',
            30: 'symmetry_worst',
            31: 'fractal_dimension_worst',
        }, inplace=True)
        
        x = data.iloc[:,2:32]
        feature_names = x.columns.values.tolist()

        training_data = data.sample(
            frac = 0.7, 
            random_state = 200
            )  # random_state = seed

        testing_data = data.drop(training_data.index)

        # divide training/testing inputs and class, split class from inputs, convert to np.array
        train_X = training_data.iloc[:, 2:32].to_numpy()
        train_Y = training_data.iloc[:, 0:2].to_numpy()
        test_X = testing_data.iloc[:, 2:32].to_numpy()
        test_Y = testing_data.iloc[:, 0:2].to_numpy()

        return train_X, train_Y, test_X, test_Y, feature_names

    def normalise(self, data):

        data = data.copy()

        min_scaler = 0.8
        max_scaler = 1.2

        for i in data.columns:
            # min max using proportional population scalers 1.2 and 0.8
            data[i] = (data[i] - (data[i].min() * min_scaler) /
                       (data[i].max() * max_scaler) - (data[i].min() * min_scaler))

            # normalise between 0 and 1
            data[i] = (data[i] - data[i].min()) / \
                (data[i].max() - data[i].min())

        return data

class MultiLayerPerceptron:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        print('\n---------------------------------------------------------------------------------\n---------------------- MULTILAYER PERCEPTRON MODEL LOADING ----------------------\n')
        
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y 
        
        start = time.time()
        self.training()
        self.testing()
        finish = time.time()
        self.time_taken = finish - start

    def training(self):
        model = Sequential()
        
        model.add(Dense(units = 50, 
                        kernel_initializer = 'uniform', 
                        activation = 'swish', 
                        input_dim = 30
                        ))
        
        model.add(Dropout(rate = 0.1))
        
        model.add(Dense(units = 25, 
                        kernel_initializer = 'random_normal', 
                        activation = 'swish', 
                        ))
        
        model.add(Dropout(rate = 0.1))

        model.add(Dense(units = 2, 
                        kernel_initializer = 'random_normal', 
                        activation = 'softmax'
                        ))
        
        model.compile(optimizer = 'adam', 
                      loss = 'mean_squared_error', 
                      metrics = ['accuracy'
                      ])
        
        history = model.fit(self.train_X, 
                  self.train_Y,
                  batch_size = 100, 
                  epochs = 1500
                  )
        
        self.model_complete = model

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['loss'])
        plt.title('MLP Model Performance')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Accuracy', 'Loss'], loc='best')
        plt.savefig('MLP Accuracy Loss Plot.png')
        plt.clf()

        print('\n-------------------- MULTILAYER PERCEPTRON TRAINING COMPLETE --------------------\n')
        return None

    def testing(self):
        predictions = self.model_complete.predict(self.test_X)
        cf_matrix = confusion_matrix(self.test_Y.argmax(axis = 1), predictions.argmax(axis = 1))
        tn, fp, fn, tp = cf_matrix.ravel()

        print('-- MLP MODEL ACCURACY: {} %'.format(np.round((tn + tp) / 
                                                                          (tn + fp + fn + tp) * 100)))

        #malignant prediction and incorrect / specificity
        print('-- MLP MODEL PRECISION: {} %'.format(np.round(tp / 
                                                                   (tp + fp) * 100)))                                                                          
        
        #probability that a malignant diagnosis is correct / sensitivity
        print('-- MLP MODEL TP RATE / RECALL: {} %'.format(np.round(tp / 
                                                                 (tp + fn) * 100)))
        
        #malignant prediction and incorrect / specificity
        print('-- MLP MODEL FP RATE / FALL OUT: {} %\n'.format(np.round(fn / 
                                                                   (fn + tp) * 100)))
        
        sb.heatmap(cf_matrix, annot = True, cbar = False, fmt = 'g')
        plt.savefig('MLP Model Confusion Matrix.png')
        plt.clf()

        print('------------- MULTILAYER PERCEPTRON PERFORMANCE EVALUATION COMPLETE -------------\n---------------------------------------------------------------------------------')
        return None

class SupportVectorMachine:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        print('\n---------------------------------------------------------------------------------\n---------------------- SUPPORT VECTOR MACHINE MODEL LOADING ---------------------')
        w.filterwarnings(action = 'ignore', category = DataConversionWarning)

        self.train_X = train_X
        self.train_Y = train_Y[:,0:1]
        self.test_X = test_X
        self.test_Y = test_Y [:,0:1]

        start = time.time()
        self.training()
        self.testing()
        finish = time.time()
        self.time_taken = finish - start

    def training(self):
        svc = SVC(
            kernel = 'linear', 
            C = 1
            )

        model = Pipeline([
            ('reduce_dim', PCA()),
            ('svc', svc)
        ])

        model.fit(
            self.train_X, 
            self.train_Y
            )
        
        self.model_complete = model

        print('\n-------------------- SUPPORT VECTOR MACHINE TRAINING COMPLETE -------------------\n')
        return None

    def testing(self):
        predictions = self.model_complete.predict(self.test_X)
        
        print('-- SVM MODEL ACCURACY: ', np.round(accuracy_score(self.test_Y, predictions) * 100), '%')
        print('-- SVM MODEL PRECISION: ', np.round(precision_score(self.test_Y, predictions) * 100), '%')
        print('-- SVM MODEL RECALL: ', np.round(recall_score(self.test_Y, predictions) * 100), '%')
        print('-- SVM MODEL REPORT: \n\n', classification_report(self.test_Y, predictions, target_names = ['malignant', 'benign']))
        
        sb.heatmap(confusion_matrix(self.test_Y, predictions), annot = True, cbar = False, fmt = 'g')
        plt.savefig('SVM Model Confusion Matrix.png')
        plt.clf()

        print('------------- SUPPORT VECTOR MACHINE PERFORMANCE EVALUATION COMPLETE ------------\n---------------------------------------------------------------------------------')
        return None

class DecisionTree:
    def __init__(self, train_X, train_Y, test_X, test_Y, feature_names):
        print('\n---------------------------------------------------------------------------------\n-------------------------- DECISION TREE MODEL LOADING --------------------------')

        self.train_X = train_X
        self.train_Y = train_Y[:,0:1] 
        self.test_X = test_X
        self.test_Y = test_Y[:,0:1]
        self.feature_names = feature_names
        
        start = time.time()
        self.training()
        self.testing()
        finish = time.time()
        self.time_taken = finish - start

    def training(self):
        model = DecisionTreeClassifier(
            criterion = 'entropy',
            max_depth = None, 
            ccp_alpha = 0.01
            )

        model.fit(
            self.train_X, 
            self.train_Y
            )
        
        self.model_complete = model

        print('\n------------------------ DECISION TREE TRAINING COMPLETE ------------------------\n')
        return None

    def testing(self):
        predictions = self.model_complete.predict(self.test_X)

        print('-- DT MODEL ACCURACY: ', np.round(accuracy_score(self.test_Y, predictions) * 100), '%')
        print('-- DT MODEL PRECISION: ', np.round(precision_score(self.test_Y, predictions) * 100), '%')
        print('-- DT MODEL RECALL: ', np.round(recall_score(self.test_Y, predictions) * 100), '%')
        print('-- DT MODEL REPORT: \n\n', classification_report(self.test_Y, predictions, target_names = ['malignant', 'benign']))
        
        fig = plt.figure(figsize = (12, 10))
        _ = plot_tree(self.model_complete,
                      feature_names = self.feature_names,
                      class_names = {0:'malignant', 1:'benign'},
                      filled = True,
                      fontsize = 12)
        plt.savefig('DT Model.png')
        plt.clf()

        sb.heatmap(confusion_matrix(self.test_Y, predictions), annot = True, cbar = False, fmt = 'g')
        plt.savefig('DT Model Confusion Matrix.png')
        plt.clf()

        print('----------------- DECISION TREE PERFORMANCE EVALUATION COMPLETE -----------------\n---------------------------------------------------------------------------------')
        return None

class RandomForest:
    def __init__(self, train_X, train_Y, test_X, test_Y, feature_names):
        print('\n---------------------------------------------------------------------------------\n-------------------------- RANDOM FOREST MODEL LOADING --------------------------')

        self.train_X = train_X
        self.train_Y = train_Y[:,0:1] 
        self.test_X = test_X
        self.test_Y = test_Y[:,0:1]
        self.feature_names = feature_names
        
        start = time.time()
        self.training()
        self.testing()
        finish = time.time()
        self.time_taken = finish - start

    def training(self):
        rf = RandomForestClassifier()

        model = Pipeline([
            ('reduce_dim', PCA()),
            ('rf', rf)
        ])

        model.fit(
            self.train_X, 
            self.train_Y
            )
        
        self.model_complete = model

        print('\n------------------------ RANDOM FOREST TRAINING COMPLETE ------------------------\n')
        return None

    def testing(self):
        predictions = self.model_complete.predict(self.test_X)

        print('-- RF MODEL ACCURACY: ', np.round(accuracy_score(self.test_Y, predictions) * 100), '%')
        print('-- RF MODEL PRECISION: ', np.round(precision_score(self.test_Y, predictions) * 100), '%')
        print('-- RF MODEL RECALL: ', np.round(recall_score(self.test_Y, predictions) * 100), '%')
        print('-- RF MODEL REPORT: \n\n', classification_report(self.test_Y, predictions, target_names = ['malignant', 'benign']))

        sb.heatmap(confusion_matrix(self.test_Y, predictions), annot = True, cbar = False, fmt = 'g')
        plt.savefig('RF Model Confusion Matrix.png')
        plt.clf()

        print('----------------- RANDOM FOREST PERFORMANCE EVALUATION COMPLETE -----------------\n---------------------------------------------------------------------------------')
        return None

# MAIN
def main():
    #Prep_obj = Preprocess(str(sys.argv[1]))
    Prep_obj = Preprocess('breast_cancer.csv')

    # MODELS
    MLP_Obj = MultiLayerPerceptron(Prep_obj.train_X, Prep_obj.train_Y, Prep_obj.test_X, Prep_obj.test_Y)
    SVM_Obj = SupportVectorMachine(Prep_obj.train_X, Prep_obj.train_Y, Prep_obj.test_X, Prep_obj.test_Y)
    DT_Obj = DecisionTree(Prep_obj.train_X, Prep_obj.train_Y, Prep_obj.test_X, Prep_obj.test_Y, Prep_obj.feature_names)
    RF_Obj = RandomForest(Prep_obj.train_X, Prep_obj.train_Y, Prep_obj.test_X, Prep_obj.test_Y, Prep_obj.feature_names)

    print('\nMLP TIME: ', MLP_Obj.time_taken, '(s)')
    print('\nSVM TIME: ', SVM_Obj.time_taken, '(s)')
    print('\nDT TIME: ', DT_Obj.time_taken, '(s)')
    print('\nRF TIME: ', RF_Obj.time_taken, '(s)')

if __name__ == '__main__':
    main()