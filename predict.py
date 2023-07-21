#%%
import torch
from model import network
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

test_data = pd.read_csv('/Users/ray/Desktop/educations/nn-modual-checkpoint/dataset/creditcard_test.csv', index_col=0)
test_feature = test_data[test_data.columns[:30]].values
test_label = test_data[test_data.columns[-1]].values

test_feature = torch.tensor(test_feature, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype =torch.float32).unsqueeze(1)


model = network()  # Instantiate your model
#Load weight
model.load_state_dict(torch.load("model_weight_Adam_test_fro_6000_epoch.pth"))
model.eval()
predictions, labels = [], []
with torch.no_grad():

    threshold = 0.4
    out = model(test_feature)
    pred = torch.sigmoid(out)
    pred = (out >= threshold).squeeze().numpy()
    print("pred values:", pred)

    test_accuracy = (pred == test_label.squeeze().numpy()).mean()
    formatted_percentage = "{:.2f}%".format(test_accuracy*100)
    print(f"Test Accuracy: {formatted_percentage}")


#%%
pred  = pred.tolist()
test_label.squeeze().tolist()
#

# %%
cm = confusion_matrix(pred, test_label)
cm
#%%
# Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

precision = precision_score(pred, test_label)
recall = recall_score(pred, test_label)

print(f'Precision: {precision}')
print(f'Recall: {recall}')


