import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

        # Переименование столбцов
        data = data.rename(columns={
            'Air temperature [K]': 'Air_temperature_K',
            'Process temperature [K]': 'Process_temperature_K',
            'Rotational speed [rpm]': 'Rotational_speed_rpm',
            'Torque [Nm]': 'Torque_Nm',
            'Tool wear [min]': 'Tool_wear_min'
        })

        # Преобразование категориальных данных
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Масштабирование
        numerical_features = [
            'Air_temperature_K',
            'Process_temperature_K',
            'Rotational_speed_rpm',
            'Torque_Nm',
            'Tool_wear_min'
        ]
        scaler = StandardScaler()
        data[numerical_features] = pd.DataFrame(
            scaler.fit_transform(data[numerical_features]),
            columns=numerical_features
        )

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Инициализация моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }

        # Обучение моделей
        for name, model in models.items():
            model.fit(X_train, y_train)

        # Функция оценки
        def evaluate_model(model, X_test, y_test):
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model,
                                                                       "predict_proba") else model.decision_function(
                    X_test)
            except NotFittedError:
                raise ValueError(f"Модель {model.__class__.__name__} не обучена!")

            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{model.__class__.__name__} (AUC={roc_auc:.2f})")

            return accuracy, conf_matrix, class_report, roc_auc

        # Сохранение метрик
        results = {}
        plt.figure(figsize=(10, 8))

        # Оценка и вывод результатов
        for name, model in models.items():
            st.markdown(f"### {name}")

            try:
                accuracy, conf_matrix, class_report, roc_auc = evaluate_model(model, X_test, y_test)
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
                continue

            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**ROC-AUC:** {roc_auc:.2f}")

            # Матрица ошибок
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            # Classification Report
            st.write("**Classification Report:**")
            st.code(class_report)

            results[name] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc
            }

        # Сохранение метрик для презентации
        st.session_state['model_metrics'] = results

        # Визуализация ROC-кривых
        st.subheader("ROC-кривые")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Сравнение моделей')
        plt.legend()
        st.pyplot(plt.gcf())

        # Форма для предсказаний
        st.header("Прогнозирование")
        with st.form("prediction_form"):
            st.write("Введите параметры оборудования:")
            air_temp = st.number_input("Температура воздуха (K)", value=300.0)
            process_temp = st.number_input("Температура процесса (K)", value=310.0)
            rotational_speed = st.number_input("Скорость вращения (rpm)", value=1500)
            torque = st.number_input("Крутящий момент (Nm)", value=40.0)
            tool_wear = st.number_input("Износ инструмента (мин)", value=100)
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])

            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                input_data = pd.DataFrame({
                    'Type': [product_type],
                    'Air_temperature_K': [air_temp],
                    'Process_temperature_K': [process_temp],
                    'Rotational_speed_rpm': [rotational_speed],
                    'Torque_Nm': [torque],
                    'Tool_wear_min': [tool_wear]
                })

                # Преобразование входных данных
                input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                # Прогнозы для всех моделей
                st.subheader("Результаты прогнозирования")
                for name, model in models.items():
                    try:
                        prediction = model.predict(input_data)
                        proba = model.predict_proba(input_data)[:, 1][0] if hasattr(model, "predict_proba") else \
                        model.decision_function(input_data)[0]
                        st.write(
                            f"**{name}:** {'Отказ 🔴' if prediction[0] == 1 else 'Норма 🟢'} (Вероятность: {proba:.2f})")
                    except Exception as e:
                        st.error(f"Ошибка в модели {name}: {str(e)}")