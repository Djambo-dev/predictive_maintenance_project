import streamlit as st
from reveal_slides import slides


def presentation_page():
    st.title("Презентация проекта")

    if 'model_metrics' not in st.session_state:
        st.warning("⚠️ Сначала обучите модели на странице 'Анализ и модель'.")
        return

    metrics = st.session_state['model_metrics']

    # Формирование динамической таблицы
    slides_content = """
    ## Прогнозирование отказов оборудования

    ---

    ### Результаты сравнения моделей
    | Модель              | Accuracy | ROC-AUC |
    |---------------------|----------|---------|"""

    for model_name, model_metrics in metrics.items():
        slides_content += f"\n| {model_name} | {model_metrics['accuracy']:.2f} | {model_metrics['roc_auc']:.2f} |"

    slides_content += """
    ---

    ### Выводы
    - **Лучшая модель**: Random Forest
    - **Причины**:
        - Наивысшие Accuracy и ROC-AUC
        - Минимальные ложные отрицательные прогнозы
    """

    # Отображение слайдов
    slides(
        slides_content,
        height=500,
        theme="night",
        config={"transition": "slide"}
    )