

import modelApi
import models
import pandas

model = models.TransformerModel
modelClassifier = modelApi.ModelClassifier(model = model,name="model_help_first")

data = pandas.read_csv("all_messages.csv")
print(data['data'][0])
modelClassifier.train(data,'data','target')
#print(modelClassifier.predict(data['data'][0]))


