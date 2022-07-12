import mlflow
logged_model = 'wasbs://artifacts@wnsmlopstrgci.blob.core.windows.net/26/c0e291cc85ff4256ad471a6bbcd142ca/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = [[2, 9, 6]]

x = loaded_model.predict(pd.DataFrame(data))

print(x)
print("done")