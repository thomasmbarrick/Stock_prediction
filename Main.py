import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class Predictor:
    # Functions set for organization
    def prepare_data(self, df, forecast_col, forecast_out, test_size):
        label = df[forecast_col].shift(-forecast_out)
        df['label'] = label  # Adding label column to DataFrame
        df.dropna(inplace=True)  # Dropping NaN values after shifting label
        x = np.array(df[[forecast_col]])
        x = preprocessing.scale(x)
        y = np.array(df['label'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def setting_model(self):
        df = pd.read_csv("price.csv")
        forecast_col = 'Close'
        forecast_out = 5
        test_size = 0.2
        X_train, X_test, Y_train, Y_test = self.prepare_data(df, forecast_col, forecast_out, test_size)
        learner = LinearRegression() 
        learner.fit(X_train, Y_train)
        return learner, X_test, Y_test
    
    def predict(self):
        learner, X_test, Y_test = self.setting_model()
        score = learner.score(X_test, Y_test)
        forecast = learner.predict(X_test)
        response = {'test_score': score, 'forecast_set': forecast}
        print(response)

# Instantiate the Predictor class and call the predict method
Predictor().predict()