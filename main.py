import numpy as np


def generate_house_data(num_samples: int) -> np.ndarray:
    # Generates synthetic house data with 10 features and price
    square_footage = np.random.randint(500, 5000, num_samples)
    bedrooms = np.random.randint(1, 10, num_samples)
    bathrooms = np.random.randint(1, 5, num_samples)
    age = np.random.randint(0, 100, num_samples)
    distance_to_city = np.random.uniform(0, 50, num_samples)
    floors = np.random.randint(1, 4, num_samples)
    garage = np.random.randint(0, 2, num_samples)
    garden_size = np.random.randint(0, 2000, num_samples)
    crime_rate = np.random.uniform(0, 10, num_samples)
    school_rating = np.random.uniform(1, 5, num_samples)

    # Simple formula to calculate price
    price = (square_footage * 50) + (bedrooms * 10000) + (bathrooms * 5000) - (age * 100) \
            - (distance_to_city * 200) + (floors * 5000) + (garage * 5000) \
            + (garden_size * 10) - (crime_rate * 1000) + (school_rating * 2000)

    data = np.column_stack((square_footage, bedrooms, bathrooms, age, distance_to_city,
                            floors, garage, garden_size, crime_rate, school_rating, price))

    return data


def normalize_data(data: np.ndarray) -> np.ndarray:
    # Min-Max normalization: (X - min) / (max - min)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def calculate_variance(data: np.ndarray) -> np.ndarray:
    # Calculates the variance of each feature in the dataset
    variances = np.var(data, axis=0)
    return variances


def initialize_network(input_size: int, hidden_size: int, output_size: int) -> dict:
    """
    Initializes the weights and biases of a neural network.

    Parameters:
    input_size (int): The number of neurons in the input layer.
    hidden_size (int): The number of neurons in the hidden layer.
    output_size (int): The number of neurons in the output layer.

    Returns:
    dict: A dictionary containing the weights and biases of each layer.
    """
    network = {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,  # small random weights
        "b1": np.zeros((1, hidden_size)),                        # zeros bias
        "W2": np.random.randn(hidden_size, output_size) * 0.01,  # small random weights
        "b2": np.zeros((1, output_size))                         # zeros bias
    }
    return network


def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies the ReLU (Rectified Linear Unit) activation function.

    Parameters:
    x (np.ndarray): Input array or matrix

    Returns:
    np.ndarray: Output after applying the ReLU function
    """
    return np.maximum(0, x)


def forward_pass(network: dict, X: np.ndarray) -> tuple:
    """
    Performs the forward pass of the neural network.

    Parameters:
    network (dict): The neural network parameters (weights and biases).
    X (np.ndarray): The input data.

    Returns:
    tuple: A tuple containing the output of each layer (hidden and output).
    """
    # Unpack the network parameters
    W1, b1, W2, b2 = network['W1'], network['b1'], network['W2'], network['b2']

    # Input to hidden layer
    hidden_input = np.dot(X, W1) + b1
    # Activation in hidden layer
    hidden_output = relu(hidden_input)

    # Input to output layer
    output_layer_input = np.dot(hidden_output, W2) + b2
    # The output layer has a linear activation (no activation function)

    return hidden_output, output_layer_input

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between the true values and predicted values.

    Parameters:
    y_true (np.ndarray): The true values.
    y_pred (np.ndarray): The predicted values.

    Returns:
    float: The calculated MSE.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def backpropagation(network: dict, X: np.ndarray, y: np.ndarray, hidden_output: np.ndarray, output: np.ndarray, learning_rate: float) -> dict:
    """
    Performs the backpropagation algorithm and updates the network's weights and biases.

    Parameters:
    network (dict): The neural network parameters (weights and biases).
    X (np.ndarray): The input data.
    y (np.ndarray): The true values.
    hidden_output (np.ndarray): The output of the hidden layer from the forward pass.
    output (np.ndarray): The predicted values from the forward pass.
    learning_rate (float): The learning rate for weight updates.

    Returns:
    dict: The updated neural network parameters.
    """
    # Unpack the network parameters
    W1, b1, W2, b2 = network['W1'], network['b1'], network['W2'], network['b2']

    # Calculate error
    error = output - y

    # Calculate derivative of loss function w.r.t W2 and b2
    dW2 = np.dot(hidden_output.T, error) / X.shape[0]
    db2 = np.sum(error, axis=0, keepdims=True) / X.shape[0]

    # Propagate error back to hidden layer
    hidden_error = np.dot(error, W2.T) * (hidden_output > 0)  # Derivative of ReLU

    # Calculate derivative of loss function w.r.t W1 and b1
    dW1 = np.dot(X.T, hidden_error) / X.shape[0]
    db1 = np.sum(hidden_error, axis=0, keepdims=True) / X.shape[0]

    # Update the weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}



def train_network(network: dict, data: np.ndarray, labels: np.ndarray, epochs: int, learning_rate: float) -> dict:
    """
    Trains the neural network using backpropagation.

    Parameters:
    network (dict): The neural network parameters (weights and biases).
    data (np.ndarray): The input data.
    labels (np.ndarray): The true values (labels).
    epochs (int): The number of times to iterate over the entire dataset.
    learning_rate (float): The learning rate for weight updates.

    Returns:
    dict: The trained neural network parameters.
    """
    for epoch in range(epochs):
        total_loss = 0

        for i in range(data.shape[0]):
            # Forward pass
            X, y = data[i:i+1], labels[i:i+1]  # Selecting one sample at a time
            hidden_output, output = forward_pass(network, X)

            # Compute loss
            loss = mean_squared_error(y, output)
            total_loss += loss

            # Backpropagation and update network
            network = backpropagation(network, X, y, hidden_output, output, learning_rate)

        avg_loss = total_loss / data.shape[0]
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    return network


def main() -> None:
    # Generating and normalizing the data
    sample_data = generate_house_data(1000)  # Generate data for 1000 houses
    normalized_data = normalize_data(sample_data[:, :-1])  # Normalize features, excluding price

    # Calculating the variance of each feature
    feature_variances = calculate_variance(normalized_data)
    print(feature_variances)  # Displaying the variance of each feature for review
    # Indexes of the selected features
    selected_feature_indexes = [0, 6, 2, 1]  # Indexes for Square Footage, Garage, Bathrooms, Bedrooms

    # Filtering the dataset to include only the selected features
    filtered_data = normalized_data[:, selected_feature_indexes]

    # Displaying the first few rows of the filtered data for review
    print(filtered_data[:5])

    # Initializing the neural network with 4 input neurons, 5 hidden neurons, and 1 output neuron
    neural_network = initialize_network(input_size=4, hidden_size=5, output_size=1)

    print(neural_network)

    # Example forward pass with a sample input (the first row of the filtered data)
    sample_input = filtered_data[0:1]
    hidden_output, output = forward_pass(neural_network, sample_input)

    print(hidden_output, output)  # Displaying the output of the hidden layer and the final output for review

    # Example usage of backpropagation (with a small learning rate)
    learning_rate = 0.01
    # Assuming we have true values (y) for the sample input
    y_true_sample = np.array([[sample_data[0, -1]]])  # The price of the first house
    #updated_network = backpropagation(neural_network, sample_input, y_true_sample, output, learning_rate)

    #print(updated_network)  # Displaying the updated network parameters for review

    labels = sample_data[:, -1].reshape(-1, 1)
    epochs = 10
    trained_network = train_network(neural_network, filtered_data, labels, epochs, learning_rate)

    print(trained_network)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
