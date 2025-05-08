import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных через интерфейс
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                              'Tool wear [min]']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели (например, Random Forest)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Визуализация результатов
        st.subheader("Результаты оценки модели")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**ROC-AUC:** {roc_auc:.2f}")

        # Матрица ошибок
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # ROC-кривая
        st.subheader("ROC-кривая")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(plt.gcf())

        # Форма для предсказаний
        st.header("Предсказание отказа")
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
                # Преобразование введенных данных
                input_data = pd.DataFrame({
                    'Type': [product_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })
                input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                # Предсказание
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[:, 1][0]
                st.success(
                    f"**Результат:** {'Отказ' if prediction[0] == 1 else 'Нет отказа'} (Вероятность: {probability:.2f})")