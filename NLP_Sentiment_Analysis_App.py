import pandas as pd
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer
from collections import Counter

# URL of the image
image_url = "https://media.istockphoto.com/id/911633218/vector/abstract-geometric-medical-cross-shape-medicine-and-science-concept-background.jpg?s=612x612&w=0&k=20&c=eYz8qm5xa5wbWCWKgjOpTamavekYv8XqPTA0MC4tHGA="

# Set the background image using custom CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("{image_url}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
# Apply the custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load csv file as df
df = pd.read_csv("C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Final_capstone_Project\\sentiment.csv")

# Load the saved model
log_reg_model = joblib.load('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Final_capstone_Project\\log_reg_model.joblib')
word2vec_model = joblib.load('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Final_capstone_Project\\word2vec_model.joblib')  # Load Word2Vec model if saved
scaler = joblib.load('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Final_capstone_Project\\scaler.joblib')

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Function to preprocess and vectorize the text
def vectorize_text(text, model):
    valid_words = [word for word in text if word in model.wv.index_to_key]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)

def preprocess_and_predict(review):
    # Tokenize and lemmatize the review
    tokens = word_tokenize(review.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Vectorize and scale
    vector = vectorize_text(lemmatized_tokens, word2vec_model)
    vector_scaled = scaler.transform(vector.reshape(1, -1))
    
    # Make prediction
    prediction = log_reg_model.predict(vector_scaled)
    sentiment_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    return sentiment_mapping[prediction[0]]


# Title and description
st.title('NLP - Sentiment Analysis on Healthcare Reviews')
# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Project Overview", "Industry Background", "Sentiment Classifier", "Model Evaluation", "Insights & Visualizations", "Challenges", "Recommendations"])

# Introduction Tab
with tab1:
    st.header("Project Overview")
    st.markdown("""
        <div style="text-align: justify;">
        <p><strong>Objective:</strong></p>
        <p>The goal of this project is to develop a model that can classify sentiments in healthcare reviews. 
        This involves analyzing text data from healthcare reviews and determining whether the sentiment expressed 
        in each review is positive, negative, or neutral.</p>

        <p><strong>Task:</strong></p>
        <ol>
            <li><strong>Data Preprocessing:</strong> This task involves cleaning and preparing the text data from healthcare reviews. 
            It includes tasks like text tokenization, removing stop words, and handling any missing data.</li>
            <li><strong>Sentiment Analysis Model:</strong> Develop a machine learning or natural language processing (NLP) model that can classify 
            sentiments in healthcare reviews. This model should be able to categorize reviews as positive, 
            negative, or neutral based on the text content.</li>
            <li><strong>Model Evaluation:</strong> Assess the performance of the sentiment analysis model using appropriate evaluation metrics. 
            This step is crucial to ensure the model's accuracy and effectiveness.</li>
            <li><strong>Insights & Visualization:</strong> After building and evaluating the model, generate insights from the sentiment analysis results. 
            Visualize the data and findings to communicate the results effectively.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("Industry Background")
    st.markdown("""
    <div style='text-align: justify;'>
        The healthcare industry is vast and complex, dealing with a massive amount of data generated daily from various sources such as patient records, 
        clinical trials, research papers, medical imaging, and patient feedback. Traditionally, this data was largely unstructured, making it challenging 
        to analyze and derive meaningful insights. However, with advancements in Natural Language Processing (NLP), the healthcare sector has seen significant
        improvements in data handling and analysis, particularly with unstructured text data. One of the applications of NLP in Healthcare is Sentiment Analysis, 
        i.e., analyzing patient reviews and feedback. Here are the key benefits of using NLP in healthcare:
        <ul>
            <li>Analyzes patient feedback from surveys, social media, and reviews to gauge satisfaction levels, helping healthcare providers identify areas needing improvement.</li>
            <li>Identifies common pain points and positive experiences, enabling providers to make informed decisions to improve patient interactions and care quality.</li>
            <li>Helps in the early identification of potential problems, such as dissatisfaction with treatment or services, allowing for timely intervention.</li>
            <li>Provides insights into patient emotions and concerns, allowing for more personalized communication and care strategies.</li>
            <li>Supports data-driven decision-making by providing a clear picture of patient sentiments, which can influence policy and operational changes.</li>
            <li>Detects trends in patient sentiment over time, helping healthcare organizations anticipate needs and adjust services accordingly.</li>
            <li>Monitors public sentiment about the healthcare organization, enabling proactive management of the brand’s reputation and addressing negative feedback swiftly.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Sentiment Classifier Tab
with tab3:
    st.header("Sentiment Classifier")
    # Input field for the new review
    user_review = st.text_input("Enter your review:")

    # Button to classify sentiment
    if st.button('Classify'):
        if user_review:
            sentiment = preprocess_and_predict(user_review)
            st.write(f'The sentiment of the review is: {sentiment}')
        else:
            st.write('Please enter a review before classifying.')

# Model Evaluation Tab
with tab4:
    st.header("Model Evaluation")
    st.write("""
    Sentiment analysis can be approached using a variety of machine learning algorithms or deep learning models. The choice of algorithm depends on factors such as the size of the dataset and the complexity of the task. 
    Some of the algorithms used are Logistic Regression, Multinomial Naive Bayes and Support Vector Machines for this task. The following table presents the metrics commonly used to evaluate these models.
""") 
    # Metrics for model
    training_data = {
        'Model': ['Logistic Regression', 'Multinomial Naive Bayes', 'Support Vector Machine'],
        'Accuracy': [1.0, 0.91, 1.0],  # Replace these with actual values
        'Precision': [1.0, 0.93, 1.0],  # Replace these with actual values
        'Recall': [1.0, 0.91, 1.0],     # Replace these with actual values
        'F1 Score': [1.0, 0.91, 1.0]    # Replace these with actual values
    }

    # Create the DataFrame
    metrics_df1 = pd.DataFrame(training_data)

    # Set the 'Model' column as the index
    metrics_df1.set_index('Model', inplace=True)

    st.write("### Model Comparison Table - Training Data")
    st.table(metrics_df1)

    # Metrics for model
    testing_data = {
        'Model': ['Logistic Regression', 'Multinomial Naive Bayes', 'Support Vector Machine'],
        'Accuracy': [1.0, 0.87, 1.0],  # Replace these with actual values
        'Precision': [1.0, 0.90, 1.0],  # Replace these with actual values
        'Recall': [1.0, 0.87, 1.0],     # Replace these with actual values
        'F1 Score': [1.0, 0.87, 1.0]    # Replace these with actual values
    }

    # Create the DataFrame
    metrics_df2 = pd.DataFrame(testing_data)

    # Set the 'Model' column as the index
    metrics_df2.set_index('Model', inplace=True)

    st.write("### Model Comparison Table - Testing Data")
    st.table(metrics_df2)

    st.write("""
             Both Logistic Regression and SVM have achieved perfect scores (1.0) across all metrics, which indicates that both models are performing consistently well on the dataset.
             Choosen Logistic Regression for its simplicity, interpretability, and scalability. It’s less likely to overfit on a small dataset and is easier implement in production.
             """)

# Insights & Visualization Tab
with tab5:
    st.subheader("Insights & Visualizations")

    # Map the numeric sentiment categories to corresponding labels
    sentiment_labels = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}
    df['Sentiment_Label'] = df['sentiment_category'].map(sentiment_labels)

    # Count the sentiment labels
    sentiment_distribution = df['sentiment_category'].value_counts()

    # Create a bar chart using Plotly
    fig = px.bar(sentiment_distribution, x=sentiment_distribution.index, y=sentiment_distribution.values,
                 labels={'x': 'Sentiment Category', 'y': 'Count'},
                 color=sentiment_distribution.index,
                 title='Sentiment Distribution')

    # Add frequency values on top of each bar
    fig.update_traces(text=sentiment_distribution.values,
                      textposition='outside')

    # Rotate x-axis labels if needed
    fig.update_layout(xaxis_tickangle=-45)

    # Customize x-axis ticks
    fig.update_xaxes(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])

    # Remove the legend
    fig.update_layout(showlegend=False)

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)
    st.info("""
        - **Neutral Sentiment (0)**: This is the most frequent sentiment category with 393 reviews, indicating a balanced view of the healthcare services by the reviewers.
        - **Negative Sentiment (-1)**: There are 387 negative reviews, which is quite close to the number of neutral reviews, suggesting some dissatisfaction among patients.
        - **Positive Sentiment (1)**: The least frequent category with 220 reviews, which might indicate areas for improvement in the service that could enhance patient satisfaction.
        """)

    # Define the confusion matrix values
    conf_matrix = np.array([[77, 0, 0],
                            [0, 79, 0],
                            [0, 0, 44]])

    # Define the x and y labels
    label1 = ['predicted -1', 'predicted 0', 'predicted 1']
    label2 = ['actual -1', 'actual 0', 'actual 1']

    # Create a heatmap using Plotly
    fig_conf_matrix = go.Figure(data=go.Heatmap(
                    z=conf_matrix,
                    x=label1,
                    y=label2,
                    colorscale='blues'
                    ))

    # Update the layout
    fig_conf_matrix.update_layout(
        title='Confusion Matrix on Testing Data',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label'),
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig_conf_matrix)

    st.info("""
        **Confusion Matrix Interpretation:**

        - **True Positives:** 
        - Class -1: 77 reviews correctly identified as negative.
        - Class 0: 79 reviews correctly identified as neutral.
        - Class 1: 44 reviews correctly identified as positive.

        - **False Positives and False Negatives:** 
        - There are no false positives or false negatives in the confusion matrix, indicating that all predictions match the actual sentiments accurately.

        - **Overall Model Performance:** 
        - The model demonstrates perfect classification performance on the testing data for all sentiment classes, with no misclassifications observed.
        """)
    # Count the occurrences of each sentiment score
    df_counts = df['sentiment_score'].value_counts().reset_index()
    df_counts.columns = ['sentiment_score', 'count']

    # Create the scatter plot
    fig = px.scatter(df_counts, x='count', y='sentiment_score', 
                    size='count',  # Optionally adjust the size of points based on count
                    title='Scatter Plot of Sentiment Scores with Counts',
                    labels={'count': 'Count', 'sentiment_score': 'Sentiment Score'},
                    color='sentiment_score', 
                    color_discrete_sequence=px.colors.qualitative.Set2)

    # Customize the scatter plot
    fig.update_layout(
        yaxis_title='Sentiment Score',
        xaxis_title='Count',
        hovermode='closest'
    )

    # Display the interactive scatter plot in Streamlit
    st.plotly_chart(fig)

    st.info("""
    - **Plot Interpretation:** 
    - This scatter plot visualizes the distribution of sentiment scores with their corresponding counts in the dataset. Each point represents a sentiment score and its frequency.
    - The x-axis shows the count of reviews for each sentiment score, while the y-axis represents the sentiment scores. The size of each point indicates the number of reviews associated with that sentiment score.
    - This plot allows us to easily observe the distribution of sentiment scores and identify which sentiment categories are most common or rare in the dataset. For example, if the points for a particular sentiment score are larger, it indicates that there are more reviews with that sentiment score.
    """)

    # Display the saved wordcloud image in Streamlit
    st.subheader('Word Cloud based on Review Text')
    st.image("C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Final_capstone_Project\\wordcloud.png", use_column_width=True)
    st.info("""
    **Word Cloud Interpretation:**
    - The Word Cloud visualizes the most frequently occurring words from the review texts. Words that appear larger in the cloud represent higher frequencies, while smaller words are less common.
    - This visualization helps in quickly identifying the key terms and themes prevalent in the reviews. Words that are prominently displayed can highlight common topics, sentiments, or areas of interest within the dataset,
    which can be valuable for qualitative analysis and further sentiment investigations
    """)
    # Count word frequencies
    # word_freq = Counter([word for tokens in df['Review_Text_lemma'] for word in tokens])

    # # Get the top 10 recurring words
    # top_10_words = word_freq.most_common(10)

    # # Convert to DataFrame for easier display
    # top_10_df = pd.DataFrame(top_10_words, columns=['Word', 'Count'])
   
    # # Display the table in Streamlit
    # st.subheader("Top 10 Recurring Words:")
    # st.dataframe(top_10_df)
    st.subheader("Top 10 Recurring Words")
    st.image("C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Final_capstone_Project\\Top_10_words.png")
    st.info("""
    **Top 10 Recurring Words Interpretation:**
    - This table displays the top 10 most frequently occurring words from the review texts. Each word is listed along with its total count of occurrences.
    - The recurring words highlight key themes and concepts frequently mentioned by reviewers. Words that appear at the top of the list are those most often used, which can provide insights into common topics and sentiments expressed in the reviews.
    - This information can be used to improve content analysis, tailor services, and address recurring issues mentioned by reviewers.
    """)
with tab6:
    st.subheader("Challenges")
    st.write("""
    - Small Dataset
    - Managing unstructured and noisy text data
    - Detecting sarcasm and nuanced sentiments
    - Handling imbalanced datasets where certain sentiment classes dominate
    - Ensuring the model is sensitive to domain-specific language and context in healthcare
    """)

with tab7:
    st.subheader("Recommendations")
    st.write("""
    Here are some recommendations to improve the sentiment analysis model and its performance:
    
    - The model works well on both training and testing data with 100% accuracy.
    - The model can be scaled to handle larger datasets.
    - More review data is required to evaluate how well the model generalizes to new data.
    - Addressing imbalanced datasets to ensure all sentiment classes are represented.
    - The model is sensitive to domain-specific language and context.
    - Encourage patients to provide reviews to enhance the dataset and improve model performance.
    - Regularly assess the model's performance on new data to ensure its accuracy and make necessary adjustments.
    - Consider using more advanced models, such as transformers (e.g., BERT or GPT), to capture complex language patterns and context better.
             
    """)