import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="PROJECT 5")

option = st.sidebar.radio("Select Option", ["HOME", "CURRENCY CONVERTER", "SUPERVISED", "UNSUPERVISED"])

def page1():
    st.title("MACHINE LEARNING NOTES")

    st.subheader("ML DEFINITION:")
    st.write("Machine learning algorithms are methods that allow computers to learn patterns from data and make decisions or predictions without being exactly programmed.")
    st.markdown("---")

    st.subheader("TYPES:")
    sel = st.selectbox("Select type", ["supervised", "unsupervised", "reinforcement"])

    if sel == "supervised":
        st.subheader("DEFINITION:")
        st.write("Supervised means learning from labelled data. Labelled data means where input and output are defined.")

        st.subheader("TECHNIQUE:")
        st.write("There are two techniques in supervised: classification and regression.")
        st.write("Classification:- In this, the computer gives only two choices e.g. 0 or 1.")
        st.write("Regression:- It is continuous value. It is used when you want to predict numbers from input or dataset.")

        st.subheader("ALGORITHMS:")
        st.write("Linear Regression- is continuous flow of data. It is used for continuous values.")
        st.write("Logistic Regression- It is used for classification problems where output is binary, like 0 or 1.")
        st.write("Decision Tree- It is a flowchart-like structure where decisions are made based on conditions at each node.")
        st.write("Random Forest- Multiple decision trees. Builds many small decision trees one after another, and each tree tries to fix mistake of previous tree.")
        st.write("Xgboost- Almost used in every task. Advanced boosting algorithm used for both classification and regression. It works by combining multiple weak models to make a strong model.")

    elif sel == "unsupervised":
        st.subheader("DEFINITION:")
        st.write("Unsupervised means learning from unlabeled data. It finds pattern in unlabelled data.")

        st.subheader("TECHNIQUE:")
        st.write("Clustering:- Grouping similar things together without labels is clustering. Clustering is used when the machine automatically finds pattern or groups in the data without being told about those groups.")
        st.write("Dimensionality Reduction:- Reduce data complexity while keeping important features. Useful for visualization or noise reduction.")
        st.write("Association Rule Learning:- Find relationships between variables in large datasets.")
        st.write("Anomaly Detection:- Find outliers or rare events without labeled data.")

        st.subheader("ALGORITHMS:")
        st.write("K-NN - In scatterplot, it creates centroid and the nearest data to centroid is grouped together and that group is called cluster.")
        st.write("K-Mean - Grouping the similar/nearest data and we have to find the mean, then the given mean goes onto the cluster.")
        st.write("DB SCAN - It combines the particular data. The data that we want is grouped and it creates multiple groups/clusters.")

    elif sel == "reinforcement":
        st.subheader("DEFINITION:")
        st.write("It works/learns from the feedback of users and from previous moves. It uses feedback in the form of reward signal to learn what action is best in a given situation.")

        st.write("IT HAS TWO TYPES:")
        st.write("Q-Learning:- It is a value-based algorithm where an agent learns the quality (Q) of actions to take in a given state.")
        st.write("SARSA (State-Action-Reward-State-Action):- Similar to Q-learning but updates Q-values using the action actually taken by the policy.")

        st.subheader("TECHNIQUE: (Reinforcement Learning doesn't use techniques the same way as supervised/unsupervised)")
        st.write("1. Value-Based:- In this, the agent tries to learn the value of each action in a state, like in Q-Learning and SARSA.")
        st.write("2. Policy-Based:- Here, the agent directly learns a policy that maps states to actions, without using a value function.")
        st.write("3. Actor-Critic:- It is a combination of both value-based and policy-based methods. One model (actor) chooses the action, and the other (critic) evaluates it.")
        st.write("4. Model-Based:- In this, the agent builds a model of the environment and uses it to plan actions.")
        st.write("5. Model-Free:- The agent learns directly from interactions with the environment without building a model.")

        st.subheader("ALGORITHMS:")
        st.write("Q-Learning:- It is a value-based algorithm where an agent learns the quality (Q) of actions to take in a given state.")
        st.write("SARSA (State-Action-Reward-State-Action):- Similar to Q-learning but updates Q-values using the action actually taken by the policy.")
        st.write("Deep Q Network (DQN):- Combines Q-learning with deep neural networks to handle large state spaces.")
        st.write("Policy Gradient:- Instead of learning value functions, it directly learns the optimal policy that maps states to actions.")
        st.write("REINFORCE Algorithm:- A basic policy gradient method which updates the policy using rewards from complete episodes.")
        st.write("Actor-Critic:- Combines value-based and policy-based methods by having two models: an actor to choose actions, and a critic to evaluate them.")
        st.write("Proximal Policy Optimization (PPO):- An advanced policy optimization algorithm that improves stability and efficiency.")
        st.write("Deep Deterministic Policy Gradient (DDPG):- Used for continuous action spaces. It combines actor-critic with deterministic policy.")
    st.markdown("---")

    st.subheader("Algorithm")
    opt = st.selectbox("Select", ["Linear", "Logistic", "Random Forest", "Decision Tree", "K-NN", "K-MEANS"])

    if opt == "Linear":
        st.write("Linear Regression Example:")
        st.write("Linear Regression:- It is used to predict continuous values using a straight line based on input features.")
        st.code("""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
        """)
        st.write("Graph:")
        st.image("linear_regression.png", caption="Linear Regression Line")

    elif opt == "Logistic":
        st.write("Logistic Regression Example:")
        st.write("Logistic Regression:- It is used for binary classification problems where the output is either 0 or 1.")
        st.code("""
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
        """)
        st.write("Graph:")
        st.image("Logistic_Regression.png", caption="Logistic Regression Classification")

    elif opt == "Random Forest":
        st.write("Random Forest Example:")
        st.write("Random Forest:- It is an ensemble of decision trees that improves accuracy by reducing overfitting.")
        st.code("""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
        """)
        st.write("Graph:")
        st.image("random_forest.png", caption="Random Forest Tree Splits")

    elif opt == "Decision Tree":
        st.write("Decision Tree Example:")
        st.write("Decision Tree:- It is a tree-like model that makes decisions based on conditions at each node.")
        st.code("""
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
        """)
        st.write("Graph:")
        st.image("Decision_Tree.png", caption="Decision Tree Diagram")

    elif opt == "K-NN":
        st.write("K-Nearest Neighbors Example:")
        st.write("K-Nearest Neighbors (K-NN):- It classifies data based on the majority label of the closest neighbors.")
        st.code("""
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
        """)
        st.write("Graph:")
        st.image("K-NN.png", caption="K-NN Classification Boundaries")

    elif opt == "K-MEANS":
        st.write("K-Means Clustering Example:")
        st.write("K-Means:- It is a clustering algorithm that groups similar data points into K clusters based on distance.")
        st.code("""
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(data)
labels = model.labels_
        """)
        st.write("Graph:")
        st.image("K-MEANS.png", caption="K-Means Clusters")
#####################################################################################3
def page2():
    st.title("CURRENCY CONVERTER")
    st.markdown("---")

    st.subheader("Select Currency and Enter Amount:")

    currencies = ["INR", "USD", "EUR", "JPY", "GBP"]

    from_currency = st.selectbox("From Currency", currencies)
    amount = st.number_input("Enter Amount", min_value=0.0, format="%.2f")
    to_currency = st.selectbox("To Currency", currencies)

    st.markdown("---")

    # Static conversion rates to INR
    rates = {
        "INR": 1.0,
        "USD": 83.0,
        "EUR": 91.0,
        "JPY": 0.57,
        "GBP": 106.0
    }

    if from_currency and to_currency:
        if from_currency == to_currency:
            converted = amount
        else:
            amount_in_inr = amount * rates[from_currency]
            converted = amount_in_inr / rates[to_currency]

        st.subheader("Converted Amount:")
        st.success(f"{amount:.2f} {from_currency} = {converted:.2f} {to_currency}")




def page3():
    st.title("SUPERVISED PAGE")
    st.markdown("---")

    # Load dataset
    try:
        df = pd.read_csv("Salary.csv")
        st.subheader("Dataset: salary.csv")
        st.dataframe(df.head())
    except FileNotFoundError:
        st.error("File salary.csv not found in directory.")
        return

    st.markdown("---")
    st.subheader("Basic Data Analysis:")
    st.write("Columns:")
    st.write(df.columns.tolist())

    st.write("Describe:")
    st.write(df.describe())

    st.write("Info:")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Column Data Types:")
    st.write(df.dtypes)


    # Backend cleanup
    df.fillna(method='ffill', inplace=True)
    st.markdown("---")
    st.subheader("Model Comparison (Regression)")

    # Split dataset (assuming simple Salary vs Experience)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    import pickle
    import io

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "K-NN": KNeighborsRegressor()
    }

    mse_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores[name] = mse

    st.write("MSE Scores:")
    st.write(mse_scores)

    best_model_name = min(mse_scores, key=mse_scores.get)
    st.bar_chart(pd.DataFrame.from_dict(mse_scores, orient='index', columns=["MSE"]))
    st.success(f"Best model based on MSE: {best_model_name}")
 
    
    st.markdown("---")
    st.subheader("Make a Prediction with Best Model")

    col1, col2 = st.columns(2)
    with col1:
        val1 = st.number_input("Enter Feature 1 (e.g. Experience):", min_value=0.0, step=0.1)
    with col2:
        val2 = st.number_input("Enter Feature 2 (e.g. Age):", min_value=0.0, step=0.1)

    if val1 and val2:
        loaded_model = pickle.load(open("best_model.pkl", "rb"))
        user_input = np.array([[val1, val2]])
        prediction = loaded_model.predict(user_input)
        st.success(f"Predicted Output: ₹ {prediction[0]:,.2f}")



def page4():
    st.title("UNSUPERVISED PAGE")
    st.markdown("---")

    st.subheader("Dataset: mall.csv")

    try:
        df = pd.read_csv("mall.csv")
        st.dataframe(df.head())
    except FileNotFoundError:
        st.error("File mall_customer.csv not found in directory.")
        return

    st.markdown("---")
    st.subheader("Data Analysis")
    st.write("Columns:")
    st.write(df.columns.tolist())
    st.write("Describe:")
    st.write(df.describe())
    st.write("Info:")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Column Data Types:")
    st.write(df.dtypes)

    # Backend cleanup
    df.fillna(method='ffill', inplace=True)

    # Assume 2 features used for clustering (customize if needed)
    X = df.iloc[:, [3, 4]]  # Example: 'Annual Income', 'Spending Score'

    st.markdown("---")
    st.subheader("Elbow Method for Optimal Clusters")

    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    sse = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Elbow Plot
    fig, ax = plt.subplots()
    ax.plot(K_range, sse, marker='o')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Sum of Squared Distances (Inertia)")
    ax.set_title("Elbow Method")
    st.pyplot(fig)
    st.success(f"Number of clusters used: {kmeans.n_clusters}")


    st.markdown("---")
    st.subheader("Predict Cluster for New Customer")

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income (k$):", min_value=0.0, step=1.0)
    with col2:
        score = st.number_input("Spending Score (1–100):", min_value=0.0, step=1.0)

    if income and score:
        new_point = np.array([[income, score]])
        cluster = kmeans.predict(new_point)[0]
        st.success(f"This customer belongs to cluster: {cluster}")



if option == "HOME":
    page1()
elif option == "CURRENCY CONVERTER":
    page2()
elif option == "SUPERVISED":
    page3()
elif option == "UNSUPERVISED":
    page4()
