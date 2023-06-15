import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Set the page title
st.title("SUV Predict")

# Load the CSV file
csv_file = "suv_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

#Global variable here
cleanDataset = st.session_state   # Store the cleaned DataFrame in session state

# Sidebar navigation
st.sidebar.title("Navigation")
options = ["EDA", "Naive Bayes Prediction", "KNN Prediction"]
selected_option = st.sidebar.radio("Go to", options)

# EDA (Exploratory Data Analysis)
if selected_option == "EDA":
    st.subheader("Exploratory Data Analysis")

    # Display the complete dataset
    st.write("Complete Dataset",df)

    # Remove the "User ID" column from the dataset
    df_imputed = df.drop(columns=["User ID"])

    # Display the filtered dataset
    st.write("Filtered Dataset (After drop User ID)", df_imputed)
    
    # Find lines with missing values
    missing_lines = df_imputed[df_imputed.isnull().any(axis=1)]
    
    # Display lines with missing values
    st.subheader("Lines with Missing Values")
    if missing_lines.empty:
        st.write("No lines with missing values found.")
    else:
        st.write(missing_lines)
    
    st.subheader("Total Count of Lines with Missing Values")
    missing_lines_count = len(missing_lines) #Count total lines with missing values
    st.write(f"Total lines with missing values: {missing_lines_count}")

    # Button to remove missing values
    if st.button("Remove Missing Values"):
        cleanDataset['df_cleaned'] = df_imputed.dropna()  # Remove rows with missing values
        
        st.subheader("Lines that have been removed due to Missing Values")
        # Display the total line count before removal
        st.write("Total lines before removal:", len(df_imputed))

        # Display the total line count after removal
        st.write("Total lines after removal:", len(cleanDataset['df_cleaned']))

        st.markdown("***")

        # Update the values in the "Gender" column - Male:1 , Female:0
        cleanDataset['df_cleaned']["Gender"] = cleanDataset['df_cleaned']["Gender"].replace({"Male": 1, "Female": 0})
        
        st.subheader("Gender replace - Male -> 1 ; Female-> 0")
        st.write(cleanDataset['df_cleaned'])

        st.markdown("***")

       # EDA Graph 1 - Pie Chart of Male vs Female
        st.subheader("EDA Graph 1: Male vs Female")

        # Create the figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        # Pie Chart 1 - Male vs Female (All rows)
        gender_counts = cleanDataset['df_cleaned']['Gender'].value_counts()
        gender_labels = ['Female', 'Male']  # Labels for the genders (0: Female, 1: Male)
        ax1.pie(gender_counts, labels=gender_labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Interested Group")

        # Pie Chart 2 - Male vs Female (Purchased: 1)
        purchased_gender_counts = cleanDataset['df_cleaned'].loc[cleanDataset['df_cleaned']['Purchased'] == 1]['Gender'].value_counts()
        ax2.pie(purchased_gender_counts, labels=gender_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Purchased Group")

        # Pie Chart 3 - Male vs Female (Purchased: 0)
        not_purchased_gender_counts = cleanDataset['df_cleaned'].loc[cleanDataset['df_cleaned']['Purchased'] == 0]['Gender'].value_counts()
        ax3.pie(not_purchased_gender_counts, labels=gender_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title("Not Purchased Group")

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.4)

        # Display the charts
        st.pyplot(fig)

        st.text("")  #Empty next line spacing

        # EDA Graph 2 - Scatter Plot of Age with Salary
        st.subheader("EDA Graph 2: Scatter Plot of Age with Salary")

        # Create the figure and axes
        fig, ax = plt.subplots()

        # Separate data for Male and Female
        male_data = cleanDataset['df_cleaned'].loc[cleanDataset['df_cleaned']['Gender'] == 1]
        female_data = cleanDataset['df_cleaned'].loc[cleanDataset['df_cleaned']['Gender'] == 0]

        # Scatter plot for Male
        ax.scatter(male_data['Age'], male_data['EstimatedSalary'], color='blue', label='Male')

        # Scatter plot for Female
        ax.scatter(female_data['Age'], female_data['EstimatedSalary'], color='red', label='Female')

        # Set labels and title
        ax.set_xlabel("Age")
        ax.set_ylabel("Salary")
        ax.set_title("Scatter Plot of Age with Salary")

        # Add legend
        ax.legend()

        # Display the plot
        st.pyplot(fig)

# Naive Bayes Prediction
elif selected_option == "Naive Bayes Prediction":
    
    st.subheader("Naive Bayes Prediction")

    if 'df_cleaned' in cleanDataset:
        #assign the varaible form session
        df_cleaned = cleanDataset['df_cleaned']

        # Set X as all columns except the last column
        X = df_cleaned.iloc[:, :-1]
        # Set y as the last column
        y = df_cleaned.iloc[:, -1]

        #Test Size 30%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.markdown("***")
        st.title("Algo 1 - Naive Bayes Evaluation")

        # Normalize the feature matrix
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the Naive Bayes model
        naive_bayes_model = GaussianNB()
        naive_bayes_model.fit(X_train, y_train)

        # Predict the target variable for the test set
        y_pred = naive_bayes_model.predict(X_test)

        # Calculate evaluation metrics
        confusion = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Display confusion matrix as decimal values
        st.subheader("Confusion Matrix Table")
        st.write(pd.DataFrame(confusion, index=['1 (Predicted)', '0 (Predicted)'], columns=['1 (Actual)', '0 (Actual)']))

        # Display confusion matrix with decimal values
        st.subheader("Confusion Matrix Value")
        tp, fp, fn, tn = confusion.ravel()
        st.write("TP:", tp, "FP:", fp)
        st.write("FN:", fn, "TN:", tn)

        # Display other evaluation metrics
        st.subheader("Evaluation Metrics")
        st.write(f"Precision: {precision:.7f}")
        st.write(f"Recall: {recall:.7f}")
        st.write(f"F1-Score: {f_score:.7f}")
        st.write(f"Accuracy: {accuracy:.7f}")
        
        st.markdown("***")
        # Prediction Start Here
        st.write("Enter your details below:")

        # Gender
        gender_labels = {0: "Female", 1: "Male"}
        # Display Male or Female
        selected_gender_label = st.radio("Gender", list(gender_labels.values()))
        # Get value as 0/1 instead of Male/Female
        gender = int(selected_gender_label == "Male")

        # Age
        age = st.slider("Age", 17, 60)

        # Estimated Salary
        salary = st.slider("Estimated Salary", 0, 150000)

        # Normalize the input data using the same scaler
        input_data = scaler.transform([[gender, age, salary]])

        prediction_labels = {0: "Not Purchased", 1: "Purchased"}

        # Make prediction
        prediction = naive_bayes_model.predict(input_data)
        predicted_label = prediction_labels[prediction[0]]

        # Display the prediction
        st.write("Prediction:", predicted_label)

        # Estimate prediction probabilities
        probabilities = naive_bayes_model.predict_proba(input_data)
        purchased_probability = probabilities[0][1]
        not_purchased_probability = 1 - purchased_probability #AKA probabilities[0][0]

        # Display prediction probabilities table
        st.subheader("Prediction Probabilities")
        probabilities_data = pd.DataFrame(
            {
                "Purchased": [purchased_probability],
                "Not Purchased": [not_purchased_probability]
            }
        )
        st.write(probabilities_data)

    else:
        st.warning("Please perform EDA and remove missing values before running the Algorithm section.")


# KNN Prediction
elif selected_option == "KNN Prediction":
    st.title("KNN Prediction")
    
    if 'df_cleaned' in cleanDataset:
        # Assign the variable from session
        df_cleaned = cleanDataset['df_cleaned']

        # Set X as all columns except the last column
        X = df_cleaned.iloc[:, :-1]
        # Set y as the last column
        y = df_cleaned.iloc[:, -1]

        # Test Size 30%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.markdown("***")
        st.title("Algo 2 - KNN Model Evaluation")

        # Normalize the feature matrix
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the KNN model
        knn_model = KNeighborsClassifier(n_neighbors=6)
        knn_model.fit(X_train, y_train)

        # Predict the target variable for the test set
        y_pred = knn_model.predict(X_test)

        # Calculate evaluation metrics
        confusion = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Display confusion matrix as decimal values
        st.subheader("Confusion Matrix Table")
        st.write(pd.DataFrame(confusion, index=['1 (Predicted)', '0 (Predicted)'], columns=['1 (Actual)', '0 (Actual)']))

        # Display confusion matrix with decimal values
        st.subheader("Confusion Matrix Value")
        tp, fp, fn, tn = confusion.ravel()
        st.write("TP:", tp, "FP:", fp)
        st.write("FN:", fn, "TN:", tn)

        # Display other evaluation metrics
        st.subheader("Evaluation Metrics")
        st.write(f"Precision: {precision:.7f}")
        st.write(f"Recall: {recall:.7f}")
        st.write(f"F1-Score: {f_score:.7f}")
        st.write(f"Accuracy: {accuracy:.7f}")

        st.markdown("***")
        # Prediction Start Here
        st.write("Enter your details below:")

        # Gender
        gender_labels = {0: "Female", 1: "Male"}
        # Display Male or Female
        selected_gender_label = st.radio("Gender", list(gender_labels.values()))
        # Get value as 0/1 instead of Male/Female
        gender = int(selected_gender_label == "Male")

        # Age
        age = st.slider("Age", 18, 60)

        # Estimated Salary
        salary = st.slider("Estimated Salary", 0, 150000)

        # Normalize the input data using the same scaler
        input_data = scaler.transform([[gender, age, salary]])

        prediction_labels = {0: "Not Purchased", 1: "Purchased"}

        # Make prediction
        prediction = knn_model.predict(input_data)
        predicted_label = prediction_labels[prediction[0]]

        # Display the prediction
        st.write("Prediction:", predicted_label)

        # Estimate prediction probabilities
        probabilities = knn_model.predict_proba(input_data)
        purchased_probability = probabilities[0][1]
        not_purchased_probability = 1 - purchased_probability #AKA probabilities[0][0]

        # Display prediction probabilities table
        st.subheader("Prediction Probabilities")
        probabilities_data = pd.DataFrame(
            {
                "Purchased": [purchased_probability],
                "Not Purchased": [not_purchased_probability]
            }
        )
        st.write(probabilities_data)

    else:
        st.warning("Please perform EDA and remove missing values before running the Algorithm section.")
