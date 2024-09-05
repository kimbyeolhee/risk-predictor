import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class TabularDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.categorical_features = ["Sex", "HTN", "DM", "DM_insulin", "Current_smoking", "Hyperlipidemia", "Chronic_kidney_disease", "COPD", "Pre_MI", "Pre_stroke", "Pre_PCI", "Pre_CABG", "BASE_ECHO", "EXT_DISEASE", "DIS_LM", "DIS_LAD", "DIS_LCX"]
        self.numerical_features = ["Age", "BMI", "Ht.", "Wt.", "Creatinine", "EF"]
        self.x_features = self.categorical_features + self.numerical_features
        self.y_feature = "Death"

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df = self._preprocess_categorical_data(df)
        df = self._preprocess_numerical_data(df)

        X, y = df[self.x_features], df[self.y_feature]

        return X, y
    
    def _preprocess_categorical_data(self, df):
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Sex 범주형 변수 처리
        ## Male, M -> 0
        ## Female, F -> 1
        df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'Male' or x == 'M' else 1)
        sex_onehot_encoded = onehot_encoder.fit_transform(df[['Sex']])
        df['Sex'] = sex_onehot_encoded

        # HTN 범주형 변수 처리
        ## 0 = yes, 1 = no, 2 = unknown
        ## NaN 값은 2으로 처리
        df['HTN'] = df['HTN'].apply(lambda x: 2 if pd.isna(x) else x)
        htn_onehot_encoded = onehot_encoder.fit_transform(df[['HTN']])
        df['HTN'] = htn_onehot_encoded

        # DM 범주형 변수 처리
        ## 0 = yes, 1 = no, 2 = unknown
        ## NaN 값은 2으로 처리
        df['DM'] = df['DM'].apply(lambda x: 2 if pd.isna(x) else x)
        dm_onehot_encoded = onehot_encoder.fit_transform(df[['DM']])
        df['DM'] = dm_onehot_encoded

        # DM_insulin 범주형 변수 처리
        ## 0 = yes, 1 = no, 2 = unknown
        ## NaN 값은 2으로 처리
        df['DM_insulin'] = df['DM_insulin'].apply(lambda x: 2 if pd.isna(x) else x)
        dm_insulin_onehot_encoded = onehot_encoder.fit_transform(df[['DM_insulin']])
        df['DM_insulin'] = dm_insulin_onehot_encoded

        # Current_smoking 범주형 변수 처리
        ## 0 = never, 1 = current, 2 = former, 3 = unknown
        ## NaN 값은 3으로 처리
        df['Current_smoking'] = df['Current_smoking'].apply(lambda x: 3 if pd.isna(x) else x)
        current_smoking_onehot_encoded = onehot_encoder.fit_transform(df[['Current_smoking']])
        df['Current_smoking'] = current_smoking_onehot_encoded

        # Hyperlipidemia 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        ## 1, 2, 3 값의 type이 int, float, str로 다양한데 이를 모두 int로 통일
        df['Hyperlipidemia'] = df['Hyperlipidemia'].apply(lambda x: 3 if pd.isna(x) else x)
        df['Hyperlipidemia'] = df['Hyperlipidemia'].astype(int)
        hyperlipidemia_onehot_encoded = onehot_encoder.fit_transform(df[['Hyperlipidemia']])
        df['Hyperlipidemia'] = hyperlipidemia_onehot_encoded

        # Chronic Kidney Disease 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        df['Chronic_kidney_disease'] = df['Chronic_kidney_disease'].apply(lambda x: 3 if pd.isna(x) else x)
        df['Chronic_kidney_disease'] = df['Chronic_kidney_disease'].astype(int)
        chronic_kidney_disease_onehot_encoded = onehot_encoder.fit_transform(df[['Chronic_kidney_disease']])
        df['Chronic_kidney_disease'] = chronic_kidney_disease_onehot_encoded

        # COPD 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        df['COPD'] = df['COPD'].apply(lambda x: 3 if pd.isna(x) else x)
        df['COPD'] = df['COPD'].astype(int)
        copd_onehot_encoded = onehot_encoder.fit_transform(df[['COPD']])
        df['COPD'] = copd_onehot_encoded

        # Pre_MI 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        df['Pre_MI'] = df['Pre_MI'].apply(lambda x: 3 if pd.isna(x) else x)
        df['Pre_MI'] = df['Pre_MI'].astype(int)
        pre_mi_onehot_encoded = onehot_encoder.fit_transform(df[['Pre_MI']])
        df['Pre_MI'] = pre_mi_onehot_encoded

        # Pre_stroke 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        df['Pre_stroke'] = df['Pre_stroke'].apply(lambda x: 3 if pd.isna(x) else x)
        df['Pre_stroke'] = df['Pre_stroke'].astype(int)
        pre_stroke_onehot_encoded = onehot_encoder.fit_transform(df[['Pre_stroke']])
        df['Pre_stroke'] = pre_stroke_onehot_encoded

        # Pre_PCI 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        df['Pre_PCI'] = df['Pre_PCI'].apply(lambda x: 3 if pd.isna(x) else x)
        df['Pre_PCI'] = df['Pre_PCI'].astype(int)
        pre_pci_onehot_encoded = onehot_encoder.fit_transform(df[['Pre_PCI']])
        df['Pre_PCI'] = pre_pci_onehot_encoded

        # Pre_CABG 범주형 변수 처리
        ## 1 = yes, 2 = no, 3 = unknown
        ## NaN 값은 3으로 처리
        df['Pre_CABG'] = df['Pre_CABG'].apply(lambda x: 3 if pd.isna(x) else x)
        df['Pre_CABG'] = df['Pre_CABG'].astype(int)
        pre_cabg_onehot_encoded = onehot_encoder.fit_transform(df[['Pre_CABG']])
        df['Pre_CABG'] = pre_cabg_onehot_encoded

        # BASE_ECHO 범주형 변수 처리
        ## 1 = Done, 2 = Not Done, 3 = Unknown
        ## NaN 값은 3으로 처리
        df['BASE_ECHO'] = df['BASE_ECHO'].apply(lambda x: 3 if pd.isna(x) else x)
        df['BASE_ECHO'] = df['BASE_ECHO'].astype(int)
        base_echo_onehot_encoded = onehot_encoder.fit_transform(df[['BASE_ECHO']])
        df['BASE_ECHO'] = base_echo_onehot_encoded

        # EXT_DISEASE 범주형 변수 처리
        ## 1VD, 2VD, 3VD, LM, LM+1VD, LM+2VD, LM+3VD, NaN
        df['EXT_DISEASE'] = df['EXT_DISEASE'].fillna('NaN')
        ext_disease_mapping = {
            'NaN': 0, '1VD': 1, '2VD': 2, '3VD': 3, 
            'LM': 4, 'LM+1VD': 5, 'LM+2VD': 6, 'LM+3VD': 7
        }
        df['EXT_DISEASE'] = df['EXT_DISEASE'].map(ext_disease_mapping)

        # DIS_LM 범주형 변수 처리
        ## YES, Yes, No, NaN
        ## YES는 Yes로 변경
        ## NaN은 'NaN'으로 변경
        df['DIS_LM'] = df['DIS_LM'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        df['DIS_LM'] = df['DIS_LM'].apply(lambda x: 'Yes' if x == 'YES' or x == 'Yes' else x)
        dis_lm_onehot_encoded = onehot_encoder.fit_transform(df[['DIS_LM']])
        df['DIS_LM'] = dis_lm_onehot_encoded

        # DIS_LAD 범주형 변수 처리
        ## YES, Yes, No, NaN
        ## YES는 Yes로 변경
        ## NaN은 'NaN'으로 변경
        df['DIS_LAD'] = df['DIS_LAD'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        df['DIS_LAD'] = df['DIS_LAD'].apply(lambda x: 'Yes' if x == 'YES' or x == 'Yes' else x)
        dis_lad_onehot_encoded = onehot_encoder.fit_transform(df[['DIS_LAD']])
        df['DIS_LAD'] = dis_lad_onehot_encoded

        # DIS_LCX 범주형 변수 처리
        ## YES, Yes, No, NaN
        ## YES는 Yes로 변경
        ## NaN은 'NaN'으로 변경
        df['DIS_LCX'] = df['DIS_LCX'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        df['DIS_LCX'] = df['DIS_LCX'].apply(lambda x: 'Yes' if x == 'YES' or x == 'Yes' else x)
        dis_lcx_onehot_encoded = onehot_encoder.fit_transform(df[['DIS_LCX']])
        df['DIS_LCX'] = dis_lcx_onehot_encoded

        # DIS_RCA 범주형 변수 처리
        ## YES, Yes, No, NaN
        ## YES는 Yes로 변경
        ## NaN은 'NaN'으로 변경
        df['DIS_RCA'] = df['DIS_RCA'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        df['DIS_RCA'] = df['DIS_RCA'].apply(lambda x: 'Yes' if x == 'YES' or x == 'Yes' else x)
        dis_rca_onehot_encoded = onehot_encoder.fit_transform(df[['DIS_RCA']])
        df['DIS_RCA'] = dis_rca_onehot_encoded

        # DIS_RAMUS
        ## YES, Yes, No, NaN
        ## YES는 Yes로 변경
        ## NaN은 'NaN'으로 변경
        df['DIS_RAMUS'] = df['DIS_RAMUS'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        df['DIS_RAMUS'] = df['DIS_RAMUS'].apply(lambda x: 'Yes' if x == 'YES' or x == 'Yes' else x)
        dis_ramus_onehot_encoded = onehot_encoder.fit_transform(df[['DIS_RAMUS']])
        df['DIS_RAMUS'] = dis_ramus_onehot_encoded

        return df

    def _preprocess_numerical_data(self, df):
        # Creatinine은 "1.3"과 같이 float가 str로 되어 있는 경우가 있음
        # ND는 Not Detected이므로 0으로 변경
        df['Creatinine'] = df['Creatinine'].apply(lambda x: 0 if x == 'ND' else x)
        df['Creatinine'] = df['Creatinine'].astype(float)

        return df
        
class TabularClassifier:
    def __init__(self, model_params=None, model_path= None):
        if model_path is None:
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **model_params)
        else:
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            self.model.load_model(model_path)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def save(self, model_path):
        self.model.save_model(model_path)
    

class XGBoostTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        

    def train(self):
        X, y = self.data_loader.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.train(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"AUC Score: {auc_score}")
        print(classification_report(y_test, y_pred))

def main():
    data_path = 'C:/Users/korea/OneDrive/바탕 화면/risk-predictor/datasets/tabular_datasets/temp_tabular_data.csv'
    dataloader = TabularDataLoader(data_path)

    # XGBoost 모델 파라미터
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1
    }

    model = TabularClassifier(model_params=params)
    trainer = XGBoostTrainer(model, dataloader)

    trainer.train()

if __name__ == "__main__":
    main()