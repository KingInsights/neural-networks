import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Step 1: List of USD Assets with their tickers
tickers = [
    "GC=F", "SI=F", "HG=F", "CL=F", "NG=F", "PL=F", "PA=F", 
    "ZW=F", "ZS=F", "ZC=F", "KC=F", "SB=F", "CT=F", "LE=F", "HE=F"
]

# Step 2: Cached function moved to top level
@st.cache_data
def load_data(ticker_symbol, cache_buster=None):
    try:
        ticker_data = yf.download(ticker_symbol, interval="1wk", period="2y")
        if ticker_data.empty:
            raise ValueError("Downloaded data is empty.")
        ticker_data.reset_index(inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(ticker_data["Date"]):
            ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])
        return ticker_data.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Failed to download data for {ticker_symbol}: {e}")
        return pd.DataFrame()

# Step 3: Streamlit UI for ticker selection
st.title("Neural Network Training for Ticker Assets")
ticker_symbol = st.selectbox("Select Ticker", tickers)

# Step 4: Button to download the data
if st.button("Download Asset"):
    # 🔥 RESET ALL RELEVANT VARIABLES
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.history = None

    # Download and save ticker data
    st.session_state.ticker_data = load_data(ticker_symbol, cache_buster=np.random.rand())
    if not st.session_state.ticker_data.empty:
        st.success(f"✅ Data for {ticker_symbol} downloaded. All previous data has been reset.")

# Step 5: Show chart if data exists
if "ticker_data" in st.session_state and not st.session_state.ticker_data.empty:
    ticker_data = st.session_state.ticker_data

    st.subheader(f"Plot of {ticker_symbol} Closing Prices (Last 2 Years)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ticker_data["Date"], ticker_data["Close"], label=f"{ticker_symbol} Price", color="orange")
    ax.set_title(f"{ticker_symbol} Weekly Closing Prices (Last 2 Years)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


    # Step 5: Create Button to generate sliding windows
    if st.button("Create Sliding Windows"):
        X = []
        y = []

        for i in range(len(ticker_data) - 6):
            X.append(ticker_data["Close"].iloc[i:i+6].values)
            y.append(ticker_data["Close"].iloc[i+6])

        X = np.array(X)
        y = np.array(y)

        st.session_state.X = X
        st.session_state.y = y

        st.write("Sliding Window Created:")
        st.write(f"X shape: {X.shape}")
        st.write(f"y shape: {y.shape}")

    # Step 6: Button to split into training and testing sets
    if st.button("Split Data into Train and Test"):
        if "X" in st.session_state and "y" in st.session_state:
            X = st.session_state.X
            y = st.session_state.y

            train_size = int(len(X) * 0.9)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]

            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.write(f"X_train shape: {X_train.shape}")
            st.write(f"y_train shape: {y_train.shape}")
            st.write(f"X_test shape: {X_test.shape}")
            st.write(f"y_test shape: {y_test.shape}")
        else:
            st.error("Please create sliding windows first!")

    import streamlit as st

st.title("⚡ Build and Display Model")

# Step 7: Button to build and display the model
build_model_button = st.button("Build and Display Model")

if build_model_button:
    # 🔥 Set random seeds here
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    # Import keras
    from tensorflow import keras
    from tensorflow.keras import layers

    # Build the model
    model = keras.Sequential([
        layers.Input(shape=(6,)),        
        layers.Dense(40, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Save model into session state
    st.session_state.model = model

    # ✅ Confirmation message
    st.success("✅ Model has been built and compiled successfully!")

    # Display model summary
    st.subheader("📋 Model Summary")
    summary_str = []
    model.summary(print_fn=lambda x: summary_str.append(x))
    st.text("\n".join(summary_str))

    # Display compile info
    st.subheader("⚙️ Model Compile Info")
    st.write("**Optimizer:** Adam")
    st.write("**Loss Function:** Mean Squared Error (MSE)")
    st.write("**Metrics:** Mean Absolute Error (MAE)")




# step train the model 

# Step 8: Button to train the model
train_model_button = st.button("Train Model")

if train_model_button:
    # Check if everything exists
    required_keys = ("X_train", "y_train", "X_test", "y_test", "model")
    if all(key in st.session_state for key in required_keys):
        # Load from session state
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        model = st.session_state.model

        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Training parameters
        epochs = 50
        batch_size = 8

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,       # Show TensorFlow output in terminal
            shuffle=False    # 🔥 Very important for repeatable results
        )

        # Update progress
        for epoch in range(epochs):
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Training... Epoch {epoch + 1}/{epochs}")

        status_text.text("✅ Training complete!")

        # Save training history
        st.session_state.history = history

        # Show final results
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]

        st.subheader("📊 Final Training Results")
        st.write(f"**Training Loss:** {final_loss:.4f}")
        st.write(f"**Training MAE:** {final_mae:.4f}")
        st.write(f"**Validation Loss:** {final_val_loss:.4f}")
        st.write(f"**Validation MAE:** {final_val_mae:.4f}")

    else:
        st.error("❌ Error: Missing X_train, y_train, X_test, y_test, or model. Build model and split data first!")



#*****************************************************************************************************************

# Step 7: Button to Predict and Plot on Naked Data
st.subheader("📈 Predict and Plot on Naked Data")

if st.button("Predict and Plot"):
    # Check if model and X_test, y_test exist
    required_keys = ("model", "X_test", "y_test")
    if all(k in st.session_state for k in required_keys):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # 🔥 Predict on test data
        y_pred = model.predict(X_test).flatten()

        # 🔥 Plot predictions vs actual values
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, label="Actual Price", marker='o')
        ax.plot(y_pred, label="Predicted Price", marker='x')
        ax.set_title(f"{ticker_symbol} Price Predictions vs Actual")
        ax.set_xlabel("Test Data Index")
        ax.set_ylabel("Ticker Price (USD)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        # 🔥 Calculate final test errors
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        st.session_state.test_mse = test_mse

        

        # 🔥 Show error metrics
        st.subheader("📊 Test Set Error Metrics")
        st.write(f"**Test MSE:** {test_mse:.4f}")
        st.write(f"**Test MAE:** {test_mae:.4f}")

       


    else:
        st.error("❌ Missing model or test data. Please train the model first!")



#***************************************************************************************************


# Step 8: Title and Button to create parameter combinations
st.subheader("🚀 Let's create a load of parameters combos")

if st.button("Combos"):
    import pandas as pd
    import itertools

    # Define possible values including layers
    epochs_list = [100]
    batch_size_list = [8, 16]
    neurons_list = [10]
    layers_list = [1, 3]  # Adding layers as a parameter (1, 2, or 3 layers)

    # 🔥 Generate all combinations using itertools.product
    combinations = list(itertools.product(epochs_list, batch_size_list, neurons_list, layers_list))

    # 🔥 Create DataFrame from combinations
    df_combos = pd.DataFrame(combinations, columns=['epochs', 'batch_size', 'neurons', 'layers'])

    # Save to session_state if needed later
    st.session_state.df_combos = df_combos

    # Only show the first 5 rows (default head)
    st.write(df_combos.head())  # Display only the first 5 rows

    # Show total number of combinations
    st.write(f"Total combinations: {len(df_combos)}")


#*********************************************************************************************


# Step 9: Title and Button to loop through combos
st.subheader("🍋 Now we have the juice, let's squeeze the lemon and let the model loop through all combos! Might take a while")

if st.button("Loop Through Combos and Find Best"):
    # Check if X_train, y_train, X_test, y_test and df_combos exist
    required_keys = ("X_train", "y_train", "X_test", "y_test", "df_combos")
    if all(k in st.session_state for k in required_keys):
        import pandas as pd
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        df_combos = st.session_state.df_combos.copy()  # work on a fresh copy

        # Add columns for results
        df_combos['loss'] = None
        df_combos['val_loss'] = None

        # Loop through each combination and train the model
        progress_bar = st.progress(0)
        for idx, row in df_combos.iterrows():
            # Build a fresh model for each combo, adding layers based on the "layers" parameter
            model = keras.Sequential([layers.Input(shape=(6,))])  # Input layer

            # Add the specified number of Dense layers
            for _ in range(row['layers']):
                model.add(layers.Dense(row['neurons'], activation='relu'))  # Neurons for each hidden layer

            # Output layer
            model.add(layers.Dense(1))  # Output layer (for regression)

            # Compile the model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=int(row['epochs']),
                batch_size=int(row['batch_size']),
                validation_data=(X_test, y_test),
                verbose=0  # Turn off verbose for speed
            )

            # Record final epoch loss values
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            # Save the loss values into the dataframe
            df_combos.at[idx, 'loss'] = final_loss
            df_combos.at[idx, 'val_loss'] = final_val_loss

            # Update progress bar
            progress_bar.progress((idx + 1) / len(df_combos))

        # Save updated df_combos back to session_state
        st.session_state.df_combos = df_combos

        # Show head of updated combos
        st.subheader("🧠 Model Training Results (First 5 Combos)")
        st.write(df_combos.head())

    else:
        st.error("❌ Missing data or parameter combos. Please make sure you've created them first!")



#***********************************************************************************


# Step 10: Title and Button to train using best parameter combo
st.subheader("🍋 Let's use the best juice droplet to train the model!")
st.write("We filter on MSE because it's a commonly used loss function for regression tasks and is particularly sensitive to large errors. This helps the model minimize bigger mistakes that could impact performance.")

if st.button("Train Model with Best Combo"):
    # Reset error metrics before training
    if 'final_loss' in st.session_state:
        del st.session_state['final_loss']
    if 'final_mae' in st.session_state:
        del st.session_state['final_mae']
    if 'final_val_loss' in st.session_state:
        del st.session_state['final_val_loss']
    if 'final_val_mae' in st.session_state:
        del st.session_state['final_val_mae']

    # Check if everything needed is in session_state
    required_keys = ("df_combos", "X_train", "y_train", "X_test", "y_test")
    if all(k in st.session_state for k in required_keys):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        df_combos = st.session_state.df_combos
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # 🔥 Find the best combo based on lowest val_loss
        # 🔥 Find the best combo based on lowest val_loss and include layers


        # 🔥 Find the best combo based on lowest val_loss and include layers
        best_combo = df_combos.sort_values('val_loss').iloc[0]

        # Include layers in the success message
        st.success(f"✅ Best combination found: Epochs={best_combo['epochs']}, Batch={best_combo['batch_size']}, Neurons={best_combo['neurons']}, Layers={best_combo['layers']}")


        
        # 🔥 Build the best model
        best_model = keras.Sequential([
            layers.Input(shape=(6,)),
            layers.Dense(int(best_combo['neurons']), activation='relu'),
            layers.Dense(1)
        ])

        # 🔥 Compile the model with 'mae' metric
        best_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # 🔥 Train the model
        history = best_model.fit(
            X_train, y_train,
            epochs=int(best_combo['epochs']),
            batch_size=int(best_combo['batch_size']),
            validation_data=(X_test, y_test),
            verbose=1  # Show training output
        )

        # Save the best model and history
        st.session_state.best_model = best_model
        st.session_state.best_history = history

        # 🔥 After training, show final metrics
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]

        st.subheader("📊 Final Training Results for Best Model")
        st.write(f"**Training Loss (MSE):** {final_loss:.4f}")
        st.write(f"**Training MAE:** {final_mae:.4f}")
        st.write(f"**Validation Loss (MSE):** {final_val_loss:.4f}")
        st.write(f"**Validation MAE:** {final_val_mae:.4f}")

    else:
        st.error("❌ Missing data or combos. Please run previous steps first!")






#***************************************************

# Step 11: Title and Button to Plot the Beast
st.subheader("🦁 Plot the Beast - Best Model Predictions vs Actual")

if st.button("Plot the Beast"):
    # Check if best_model, X_test, y_test exist
    required_keys = ("best_model", "X_test", "y_test")
    if all(k in st.session_state for k in required_keys):
        best_model = st.session_state.best_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # 🔥 Predict with best model
        y_pred_best = best_model.predict(X_test).flatten()

        # 🔥 Calculate error metrics
        final_loss, final_mae = best_model.evaluate(X_test, y_test, verbose=0)  # [0] gives MSE (loss), [1] gives MAE

        # Display the error metrics (Loss and MAE)
        st.subheader("📊 Model Error Metrics")
        st.write(f"**Final Loss (MSE):** {final_loss:.4f}")
        st.write(f"**Final MAE:** {final_mae:.4f}")

        # Assuming test_mse is already saved in session_state from earlier predictions
        initial_mse = st.session_state.test_mse

        # Calculate percentage difference in MSE
        percentage_change = ((initial_mse - final_loss) / initial_mse) * 100

        # Display the results
        st.write(f"**Initial Model MSE:** {initial_mse:.4f}")
        st.write(f"**Final Model MSE:** {final_loss:.4f}")
        st.write(f"**Percentage Improvement:** {percentage_change:.2f}%")

        # 🔥 Plot best predictions vs actual values
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, label='Actual Price', marker='o')
        ax.plot(y_pred_best, label='Best Model Predictions', marker='x')

        ax.set_title(f"{ticker_symbol} - Best Model Predictions vs Actual Prices")
        ax.set_xlabel('Test Data Index')
        ax.set_ylabel('Ticker Price (USD)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.error("❌ Missing best model or test data. Please train and select the best model first!")




    
