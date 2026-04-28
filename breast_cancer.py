import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier     
from xgboost import XGBClassifier                      
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

print(df.info())
print(df.describe())
print(df.isnull().sum())

corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(18, 15))
sns.heatmap(corr_matrix, 
            mask=mask,          
            cmap='RdBu_r',       
            vmax=1, vmin=-1,  
            annot=False,         
            linewidths=.5,       
            cbar_kws={"shrink": .8}) 

plt.title("Breat Cancer - Corr Heatmap", fontsize=20)
plt.show()


df.drop(["mean radius","mean perimeter","worst radius","worst perimeter","worst area"],axis=1,inplace=True)

df['target'] = cancer_data.target
y = df['target']
X=df.drop("target",axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=40)
my_models={
    "Logistic regression":LogisticRegression(random_state=40,max_iter=10000),
    "Random forest":RandomForestClassifier(n_estimators=100, random_state=40),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=40) 
}

process=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),X.columns)
    ]
)
my_updated_model=[]
for name,model in my_models.items():
    pipe=Pipeline(
        steps=[("preprocess",process),
               ("model",model)])
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss


    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred)

    rec = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    y_probs = pipe.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)

    loss = log_loss(y_test, y_probs)
    model_card = {
        "Model name": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC Score": auc,
        "Log Loss": loss,
        "Trained_Pipe": pipe 
    }
    
    my_updated_model.append(model_card)

results_df = pd.DataFrame(my_updated_model)

display_df = results_df.drop(columns=["Trained_Pipe"])

print("\n" + "="*50)
print("="*50)
print(display_df.to_string(index=False))
print("="*50)


best_model={}
best_score=-1
for x in my_updated_model:
    if x["Recall"] > best_score:
        best_model["Model name"]=x["Model name"]
        best_model["Recall"]=x["Recall"]
        best_score=x["Recall"]



print(f"\n🏆 Best Model: {best_model['Model name']}")
print(f"🎯 Highest Recall: {best_model['Recall']:.4f}")