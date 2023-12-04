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


def main() -> None:
    # Generating and normalizing the data
    sample_data = generate_house_data(1000)  # Generate data for 1000 houses
    normalized_data = normalize_data(sample_data[:, :-1])  # Normalize features, excluding price

    # Calculating the variance of each feature
    feature_variances = calculate_variance(normalized_data)
    print(feature_variances)  # Displaying the variance of each feature for review


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
