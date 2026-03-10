
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge # Changed from SGDRegressor


class NelsonSiegelSvensson:
    def __init__(self, beta0=0, beta1=0, beta2=0, beta3=0, tau1=1, tau2=1):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.tau1 = tau1
        self.tau2 = tau2

    def nss_curve(self, durations):
        """
        Calculates the Nelson-Siegel-Svensson (NSS) yield for given durations.

        This method uses the current parameters (beta0, beta1, beta2, beta3, tau1, tau2)
        to compute the yield curve according to the NSS formula.

        Args:
            durations (np.ndarray or scalar): A NumPy array or scalar representing the
                                             time to maturity for which to calculate the yield.

        Returns:
            np.ndarray or scalar: A NumPy array or scalar representing the estimated yield
                                  for the given durations.
        """
        term1 = self.beta0
        # Handle division by zero for durations close to 0
        # Use a small epsilon to avoid RuntimeWarning if durations includes 0 or near 0 values
        # In NSS, (1-e^(-x))/x approaches 1 as x approaches 0
        # term2, term3, term4 calculations will become stable with this check
        with np.errstate(divide='ignore', invalid='ignore'):
            x1 = np.where(durations == 0, 1e-9, durations / self.tau1)
            x2 = np.where(durations == 0, 1e-9, durations / self.tau2)

            term2 = self.beta1 * (1 - np.exp(-x1)) / x1
            term3 = self.beta2 * ((1 - np.exp(-x1)) / x1 - np.exp(-x1))
            term4 = self.beta3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))

            # For cases where durations is actually 0, (1-e^(-x))/x -> 1
            term2 = np.where(durations == 0, self.beta1, term2)
            term3 = np.where(durations == 0, 0, term3) # As (1-e^(-x))/x - e^(-x) -> 1 - 1 = 0 when x -> 0
            term4 = np.where(durations == 0, 0, term4) # Same for term4

        return term1 + term2 + term3 + term4

    def fit(self, durations, yields):
        """
        Fits the Nelson-Siegel-Svensson (NSS) model parameters to observed durations and yields.

        This method uses optimization (L-BFGS-B) to find the parameters (beta0, beta1, beta2, beta3, tau1, tau2)
        that minimize the sum of squared differences between the observed yields and the yields
        estimated by the NSS curve.

        Args:
            durations (np.ndarray): A NumPy array of observed maturities (time to maturity).
            yields (np.ndarray): A NumPy array of observed yields corresponding to the durations.

        Returns:
            scipy.optimize.OptimizeResult: An object containing the results of the optimization,
                                          including the optimized parameters.
        """
        def loss_function(params):
            # Ensure tau1 and tau2 are positive, as they represent time constants.
            # The minimize function might propose negative values, so we enforce positivity.
            if params[4] <= 0 or params[5] <= 0: # tau1, tau2
                return np.inf # Return infinite loss for invalid parameters

            self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2 = params
            estimated_yields = self.nss_curve(durations)
            # Handle potential NaNs in estimated_yields if durations was 0 initially without special handling
            if np.isnan(estimated_yields).any():
                return np.inf
            return np.sum((yields - estimated_yields) ** 2)

        # Initial guess for the parameters
        # tau1 and tau2 should typically be positive, hence the bounds.
        initial_guess = [0.02, -0.01, 0.05, -0.02, 1.0, 5.0] # More realistic initial guess
        bounds = [
            (None, None), # beta0
            (None, None), # beta1
            (None, None), # beta2
            (None, None), # beta3
            (1e-6, None), # tau1 (must be positive)
            (1e-6, None)  # tau2 (must be positive)
        ]

        # Optimize the parameters
        result = minimize(loss_function, initial_guess, method='L-BFGS-B', bounds=bounds)
        self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2 = result.x
        return result

    def estimate_yield(self, new_durations):
        """
        Estimates the yield for new durations using the fitted Nelson-Siegel-Svensson (NSS) parameters.

        This method leverages the `nss_curve` method with the parameters previously fitted
        by the `fit` method to predict yields for arbitrary new durations.

        Args:
            new_durations (np.ndarray or scalar): A NumPy array or scalar representing the
                                                  new maturities for which to estimate the yield.

        Returns:
            np.ndarray or scalar: A NumPy array or scalar representing the estimated yield
                                  for the given new durations.
        """
        return self.nss_curve(new_durations)


    def forward_curve(self, durations):
        term1 = self.beta0
        term2 = self.beta1 * (np.exp(-durations / self.tau1))
        term3 = self.beta2 * ((durations / self.tau1) * np.exp(-durations / self.tau1))
        term4 = self.beta3 * ((durations / self.tau2) * np.exp(-durations / self.tau2))
        return term1 + term2 + term3 + term4

    def plot_forward_curve(self, durations):
        forward_rates = self.forward_curve(durations)
        plt.figure(figsize=(8, 5))
        plt.plot(durations, forward_rates, color='navy', linewidth=2)
        plt.title("Forward Interest Rate Curve", loc='left', fontsize=14, fontweight='bold', fontname='Tahoma')
        plt.xlabel("Time to Maturity (Years)", fontsize=12, fontname='Tahoma')
        plt.ylabel("Forward Rate", fontsize=12, fontname='Tahoma')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()



if __name__ == "__main__":
    # Example data (replace with real data)
    durations = np.array([0.10,0.18,0.22,0.26,0.30,0.47,0.55,0.64,0.97,1.18,1.47])
    yields = np.array([0.0205,0.0244,0.0249,0.0249,0.0254,0.0238,0.0241,0.0245,0.0235,0.0235,0.0248])
    # Initialize the model
    nss_model = NelsonSiegelSvensson()

    # Fit the model to the data
    nss_model.fit(durations, yields)

    # Initialize an array of durations (fechas) and an empty list for estimated yields
    fechas = np.arange(0.0, 5.5, 0.1)

    estimated_yields = []

    # Loop over each duration in 'fechas' and estimate the yield
    for duration in fechas:
        estimated_yield = nss_model.estimate_yield(duration)
        estimated_yields.append(estimated_yield)

    # Now 'estimated_yields' contains the yield estimates for all durations in 'fechas'

    estimated_yield_TTM26 = nss_model.estimate_yield(0.45)
    estimated_yield_TTJ26 = nss_model.estimate_yield(0.75)
    estimated_yield_TTS26 = nss_model.estimate_yield(0.96)
    estimated_yield_TTD26 = nss_model.estimate_yield(1.21)

    print(f"Estimated Yield TTM26: {estimated_yield_TTM26:.3%} ")
    print(f"Estimated Yield TTJ26: {estimated_yield_TTJ26:.3%} ")
    print(f"Estimated Yield TTS26: {estimated_yield_TTS26:.3%} ")
    print(f"Estimated Yield TTD26: {estimated_yield_TTD26:.3%} ")


    # Compute Forward Rates
    fechas = np.arange(0.0, 2, 0.1)
    forward_rates = nss_model.forward_curve(fechas)


    # Plot Forward Rate Curve
    nss_model.plot_forward_curve(fechas)

    # Print sample forward rates
    for i in range(0, len(fechas), 10):
        print(f"Forward rate at {fechas[i]:.1f} years: {forward_rates[i]:.3%}")

def fitted_poly_output(all_polynomial_params, polynomial_degree, x):
  """
  Calculates the output of a fitted polynomial for given x values.

  Args:
      all_polynomial_params (np.ndarray): An array containing all polynomial parameters,
                                        where the first element is the intercept and subsequent
                                        elements are coefficients for x^1, x^2, ..., x^degree.
      polynomial_degree (int): The degree of the polynomial.
      x (np.ndarray or scalar): The input values for which to calculate the polynomial output.

  Returns:
      np.ndarray: The polynomial output for the given x values.
  """
  x_reshaped = np.array(x).reshape(-1, 1)
  # Create PolynomialFeatures object with include_bias=True to match the fitting process
  poly_features_eval = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
  # Transform x into polynomial features
  poly_x_eval = poly_features_eval.fit_transform(x_reshaped)
  # Calculate the dot product of the polynomial features and all parameters
  poly_output = np.dot(poly_x_eval, all_polynomial_params)
  return poly_output


def fit_polynomial_with_regularization(train_durations, train_yields, degree, alpha_val):
    """
    Fits a polynomial of a specified degree with L2 regularization to the training data.

    Args:
        train_durations (np.ndarray): The training durations (x-values).
        train_yields (np.ndarray): The training yields (y-values).
        degree (int): The degree of the polynomial to fit.
        alpha_val (float): The regularization strength (alpha) for Ridge regression.

    Returns:
        np.ndarray: An array containing all polynomial parameters, where the first element
                    is the intercept and subsequent elements are coefficients for x^1, x^2, ..., x^degree.
    """
    # Reshape train_durations for PolynomialFeatures
    train_durations_reshaped = train_durations.reshape(-1, 1)

    # Create PolynomialFeatures object. include_bias=True adds a column of ones
    # so that the intercept is handled by the polynomial features.
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)

    # Transform train_durations to polynomial features
    poly_train_features = poly_features.fit_transform(train_durations_reshaped)

    # Instantiate Ridge regressor with L2 penalty (Ridge regularization).
    # fit_intercept=False because PolynomialFeatures already added a bias (intercept) term.
    ridge_reg = Ridge(
        alpha=alpha_val,
        random_state=42,
        fit_intercept=False # Set to False as include_bias=True in PolynomialFeatures
    )

    # Fit the model to the polynomial features and training yields
    ridge_reg.fit(poly_train_features, train_yields)

    # The ridge_reg.coef_ now contains all coefficients, with the first element being the intercept.
    return ridge_reg.coef_

def fitted_poly_output(all_polynomial_params, polynomial_degree, x):
  """
  Calculates the output of a fitted polynomial for given x values.

  Args:
      all_polynomial_params (np.ndarray): An array containing all polynomial parameters,
                                        where the first element is the intercept and subsequent
                                        elements are coefficients for x^1, x^2, ..., x^degree.
      polynomial_degree (int): The degree of the polynomial.
      x (np.ndarray or scalar): The input values for which to calculate the polynomial output.

  Returns:
      np.ndarray: The polynomial output for the given x values.
  """
  x_reshaped = np.array(x).reshape(-1, 1)
  # Create PolynomialFeatures object with include_bias=True to match the fitting process
  poly_features_eval = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
  # Transform x into polynomial features
  poly_x_eval = poly_features_eval.fit_transform(x_reshaped)
  # Calculate the dot product of the polynomial features and all parameters
  poly_output = np.dot(poly_x_eval, all_polynomial_params)
  return poly_output

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def transform_data(durations, yields, test_size=0.2, random_state=None):
    """
    splits the data into training and test sets using random dropout.

    Args:
        durations (np.ndarray): The original durations (x-values).
        yields (np.ndarray): The original yields (y-values).
        num_to_drop (int): The number of observations to randomly drop from the original dataset.
        test_size (float or int): The proportion or absolute number of observations
                                  to use for the test set from the *remaining* data.
        random_state (int, optional): Seed for random number generation for reproducibility.

    Returns:
        tuple: (train_durations, train_yields, test_durations, test_yields, dropped_durations, dropped_yields)
               - train_durations (np.ndarray): Durations for the training set.
               - train_yields (np.ndarray): Yields for the training set.
               - test_durations (np.ndarray): Durations for the test set.
               - test_yields (np.ndarray): Yields for the test set.
               - dropped_durations (np.ndarray): Durations of the dropped observations.
               - dropped_yields (np.ndarray): Yields of the dropped observations.
    """
    combined_data = np.column_stack((durations, yields))
    n_total = len(combined_data)
    # Randomly select indices for the observations to be dropped
    # Set random_state for reproducibility of the dropped points
    np.random.seed(random_state)
    all_indices = np.arange(n_total)




    # Split the remaining data into training and test sets
    if len(combined_data) > 0:
        train_data, test_data = train_test_split(
            combined_data, test_size=test_size, random_state=random_state
        )


    train_durations = train_data[:, 0] if train_data.size > 0 else np.array([])
    train_yields = train_data[:, 1] if train_data.size > 0 else np.array([])
    test_durations = test_data[:, 0] if test_data.size > 0 else np.array([])
    test_yields = test_data[:, 1] if test_data.size > 0 else np.array([])

    return train_durations, train_yields, test_durations, test_yields


from datetime import datetime

def years_until_expiry(expiry_date_str):
  """
  Calculates the amount of time in years until a given expiry date.

  Args:
      expiry_date_str (str): The expiry date in 'YYYY-MM-DD' format.

  Returns:
      float: The time in years until expiry. Returns 0 if the expiry date
             is in the past or today.
  """
  try:
    expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
  except ValueError:
    raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.")

  today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

  if expiry_date <= today:
    return 0.0

  time_difference = expiry_date - today
  years = time_difference.days / 365.25 # Account for leap years
  return years


def plot_forward_curve_from_data(maturities, yields, plot_range_max=5.0):
    """
    Fits a Nelson-Siegel-Svensson model to given maturities and yields,
    then plots the derived forward interest rate curve.

    Args:
        maturities (np.ndarray): An array of observed maturities (time to maturity).
        yields (np.ndarray): An array of observed yields corresponding to the maturities.
        plot_range_max (float): The maximum maturity up to which to plot the forward curve.
    """
    # Initialize and fit the Nelson-Siegel-Svensson model
    nss_model = NelsonSiegelSvensson()
    nss_model.fit(maturities, yields)

    # Generate a range of durations for plotting the forward curve
    # Starting from a small non-zero value to avoid log(0) issues for forward rates if durations start at 0
    plot_durations = np.linspace(0.01, plot_range_max, 100)

    # Plot the forward curve using the model's method
    nss_model.plot_forward_curve(plot_durations)

    print("Forward curve plotted successfully.")


yields_tna = [0.0] * len(yields_tna_6mes)
for x in range(len(yields_tna_6mes)):
    yields_tna[x] = ((1+(yields_tna_6mes[x]/200))**2)-1

# Initialize arg_yields as a list of zeros with the same length as yields_tna
arg_yields = [0.0] * len(yields_tna)
for x in range(len(yields_tna)):
    arg_yields[x] = ((1 + yields_tna[x])**(1/12))-1
display(arg_yields)
durations = np.log(arg_durations)
mean_yield = np.mean(arg_yields)
std_yield = np.std(arg_yields)
yields = (arg_yields - mean_yield) / std_yield

plt.figure(figsize=(10, 6))
plt.scatter(durations, yields, color='blue', label='Actual Yields')
plt.title('Z score scale and centered Yields vs. log Durations', fontsize=14, fontweight='bold')
plt.xlabel('Durations log(Years)', fontsize=12)
plt.ylabel('Z score scaled and centered Yields', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

design_matrix = pd.DataFrame()

design_matrix['durations'] = durations

design_matrix['yields'] = yields

display(design_matrix)

train_durations, train_yields, test_durations, test_yields = transform_data(durations, yields, test_size=0.01, random_state=42)

print("Original durations length:", len(durations))
print("Original yields length:", len(yields))


print(f"\nTrain Durations (count={len(train_durations)}):")
print(train_durations)
print(f"Train Yields (count={len(train_yields)}):")
print(train_yields)

print(f"\nTest Durations (count={len(test_durations)}):")
print(test_durations)
print(f"Test Yields (count={len(test_yields)}):")
print(test_yields)

print(f"\nTotal observations (Train + Test): {len(train_durations) + len(test_durations)}")


degree = 5
alpha_val = 0.004*(degree**2)
polynomial_coefficients = fit_polynomial_with_regularization(train_durations=train_durations, train_yields=train_yields, degree=degree, alpha_val=alpha_val)

print(polynomial_coefficients)

# Generate a smooth range of x values for plotting from 0 to 2
x_poly_plot = np.linspace(-4, 4, 100).reshape(-1, 1)

# x_poly_plot = tuple(x_poly_plot) + tuple(train_durations)


y_poly_plot = fitted_poly_output(polynomial_coefficients, degree, list(x_poly_plot))
display(x_poly_plot.shape)


display(y_poly_plot.shape)

import matplotlib.pyplot as plt

# Create a scatter plot of the original data
plt.figure(figsize=(10, 6))
plt.scatter(train_durations, train_yields, color='blue', label='Actual Yields')
plt.scatter(test_durations, test_yields, color='red', label='test_set')
# Plot the fitted polynomial curve
# Ensure x_poly_plot and y_poly_plot are correctly defined from previous steps
if 'x_poly_plot' in locals() and 'y_poly_plot' in locals():
    plt.plot(x_poly_plot, y_poly_plot, color='red', linestyle='-', label=f'Fitted Polynomial Curve (Degree {degree})')
else:
    print("Warning: x_poly_plot or y_poly_plot not found. Cannot plot polynomial curve.")

# Add labels, title, and legend
plt.title('Actual Yields vs. Fitted Polynomial Curve', fontsize=14, fontweight='bold')
plt.xlabel('Durations log(Years)', fontsize=12)
plt.ylabel('Z score scaled and centered Yields', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Set y-axis limits as requested by the user
plt.ylim(-4,3)

plt.show()

# Example expiry date for the zero-coupon bond
expiry_date_for_prediction = "2026-03-15" # Changed to a future date for a meaningful prediction

# Calculate years until expiry using the helper function
years_to_maturity = years_until_expiry(expiry_date_for_prediction)

# Log transform the duration, as the polynomial was fitted on log durations
# Handle case where years_to_maturity might be 0 or very close to it, which would cause log(0) to be undefined.
# For a zero duration, log(0) is not applicable, but in financial context it means instantaneous.
# For prediction, we might want to avoid log(0) if it's truly 0, or handle it as a very small number.
# Let's assume for prediction, we're interested in future maturities, so years_to_maturity > 0.
if years_to_maturity <= 0:
    print(f"Warning: Expiry date {expiry_date_for_prediction} is today or in the past. Predicted yield will be 0.")
    predicted_yield_z_scaled = 0.0 # Or handle as per business logic
    predicted_actual_yield = (predicted_yield_z_scaled * std_yield) + mean_yield
    log_duration_for_prediction = np.nan # Initialize for printing
else:
    log_duration_for_prediction = np.log(years_to_maturity)

    # Predict the Z-score scaled yield using the fitted polynomial
    # 'polynomial_coefficients' and 'degree' are available from previous cells.
    predicted_yield_z_scaled = fitted_poly_output(polynomial_coefficients, degree, log_duration_for_prediction)[0]

    # Inverse transform the Z-score scaled yield to get the actual yield
    # 'mean_yield' and 'std_yield' are available from previous cells (qUD46n8RSDi0).
    predicted_actual_yield = (predicted_yield_z_scaled * std_yield) + mean_yield


print(f"For a zero-coupon bond expiring on {expiry_date_for_prediction}:")
print(f"Time until expiry: {years_to_maturity:.2f} years")
print(f"Log-transformed duration: {log_duration_for_prediction:.2f}")
print(f"Predicted Z-score scaled yield: {predicted_yield_z_scaled:.4f}")
print(f"Predicted actual yield: {predicted_actual_yield:.3%}")

import matplotlib.pyplot as plt

# 1. Get original durations and yields from kernel state
# arg_durations and arg_yields are available from previous cells (qUD46n8RSDi0)
# mean_yield and std_yield are also available for inverse scaling

# 2. Generate a continuous range of original maturities for plotting the prediction line
# Determine the range from original arg_durations
min_original_duration = np.min(arg_durations)
max_original_duration = np.max(arg_durations)

# Create a smooth, denser range of durations for plotting the continuous curve
smooth_original_durations = np.linspace(min_original_duration, max_original_duration, 100)

# Log transform these smooth durations
smooth_log_durations = np.log(smooth_original_durations)

# Predict the Z-score scaled yields for these smooth log durations
smooth_predicted_yields_z_scaled = fitted_poly_output(polynomial_coefficients, degree, smooth_log_durations)

# Inverse transform the Z-score scaled yields to get the actual predicted yields
smooth_predicted_actual_yields = (smooth_predicted_yields_z_scaled * std_yield) + mean_yield

# Reshape to 1D array if necessary
smooth_predicted_actual_yields = smooth_predicted_actual_yields.flatten()

# 3. Plotting
plt.figure(figsize=(10, 6))

# Plot original observed values as scatter points
plt.scatter(arg_durations, arg_yields, color='blue', label='Observed Yields')

# Plot the continuous fitted polynomial curve (predictions) as a line
plt.plot(smooth_original_durations, smooth_predicted_actual_yields, color='green', linestyle='-', label=f'Fitted Polynomial Curve (Degree {degree})')

# Plot the single prediction from the user-specified expiry date
# These variables (years_to_maturity, predicted_actual_yield) are from cell 'voVdU2tfXoL_'
plt.scatter(years_to_maturity, predicted_actual_yield, color='red', marker='o', s=150, label=f'Single Prediction ({expiry_date_for_prediction})', zorder=5)

# Add labels, title, and legend
plt.title('Observed vs. Predicted Yields on Original Scale', fontsize=14, fontweight='bold')
plt.xlabel('Original Maturities (Years)', fontsize=12)
plt.ylabel('Original Yields', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()
print(predicted_actual_yield)

nss_spot_rates = nss_model.estimate_yield(smooth_original_durations)
nss_maturity_prediction = nss_model.estimate_yield(years_to_maturity)
print(nss_maturity_prediction)
plt.scatter(arg_durations, arg_yields, color='blue', label='Observed Yields')
plt.scatter(years_to_maturity, nss_maturity_prediction, color='red', marker='o', s=150, label=f'Single Prediction ({expiry_date_for_prediction})', zorder=5)
plt.plot(smooth_original_durations, nss_spot_rates, color='green', linestyle='-')
# Add labels, title, and legend
plt.title('Observed vs. Predicted Yields on Original Scale with NSS', fontsize=14, fontweight='bold')
plt.xlabel('Original Maturities (Years)', fontsize=12)
plt.ylabel('Original Yields', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

import numpy as np
import matplotlib.pyplot as plt



# Example Usage with the existing arg_durations and arg_yields:
# Ensure arg_durations and arg_yields are defined (from cell qUD46n8RSDi0)
if 'arg_durations' in locals() and 'arg_yields' in locals():
    print("Plotting forward curve using existin NSS")
    plot_forward_curve_from_data(smooth_original_durations, nss_spot_rates)
else:
    print("Warning: 'arg_durations' or 'arg_yields' not found. Cannot plot forward curve.")


plot_forward_curve_from_data(smooth_original_durations, smooth_predicted_actual_yields)