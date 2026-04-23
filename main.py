import pandas as pd
import numpy as np

from utils.profiler import detect_data_type, detect_query_type, get_applicable_mechanisms
from utils.evaluator import (
    evaluate_numerical_mechanisms,
    evaluate_binary_mechanism,
    evaluate_selection_mechanism
)
from utils.visualizer import plot_error_vs_epsilon, plot_mechanism_comparison


def dp_recommendation_system(data, query, epsilon):

    # Step 1: Profiling
    data_type = detect_data_type(data)
    query_type = detect_query_type(query)
    mechanisms = get_applicable_mechanisms(data_type, query_type)

    print("Data Type:", data_type)
    print("Query Type:", query_type)
    print("Applicable Mechanisms:", mechanisms)

    #  NUMERICAL 
    if query_type == "numerical":
        true_value, results = evaluate_numerical_mechanisms(data, epsilon)

    #  BINARY 
    elif query_type == "binary":
        true_value, results = evaluate_binary_mechanism(data, epsilon)

    #  SELECTION 
    elif query_type == "selection":
        true_value, results = evaluate_selection_mechanism(data, epsilon)

    else:
        print("Unsupported query type")
        return None, None

    #  OUTPUT 
    print("\nTrue Value:", true_value)

    for mech in results:
        print(f"{mech} → Value:", results[mech].get("noisy_value", results[mech].get("selected_value")),
              "| Error:", results[mech]["error"],
              "| Score:", results[mech]["score"])

    best_mech = max(results, key=lambda m: results[m]["score"])

    print("\n Recommended Mechanism:", best_mech)

    return best_mech, results


# LOAD REAL DATASET

df = pd.read_excel("student_exam_scores_dataset.xlsx")

# Auto-detect numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
print("Numeric columns:", numeric_cols)

# Pick first numeric column
column_name = numeric_cols[0]
print("Using column:", column_name)

# Extract data
data = df[column_name].dropna().values

print("Dataset loaded. Size:", len(data))

# MULTI-EPSILON ANALYSIS (WITH VISUALIZATION)

query = "mean"
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]

laplace_errors = []
gaussian_errors = []

print("\n========== MULTI-EPSILON ANALYSIS ==========")

for eps in epsilons:
    print("\n------------------------------")
    print(f"Epsilon = {eps}")
    print("------------------------------")

    best_mech, results = dp_recommendation_system(data, query, eps)

    # Store errors for graph
    if "laplace" in results:
        laplace_errors.append(results["laplace"]["error"])
    if "gaussian" in results:
        gaussian_errors.append(results["gaussian"]["error"])


# PLOTS

plot_error_vs_epsilon(epsilons, laplace_errors, gaussian_errors)

_, results = dp_recommendation_system(data, query, 1.0)
plot_mechanism_comparison(results)

#  TEST CASES

epsilon = 1.0

print("\n========== TEST CASES ==========")

print("\n--- NUMERICAL ---")
data1 = np.array([10, 20, 30, 40, 50])
dp_recommendation_system(data1, "mean", epsilon)

print("\n--- BINARY ---")
data2 = np.array([1, 0, 1, 1, 0, 1, 0])
dp_recommendation_system(data2, "binary", epsilon)

print("\n--- SELECTION ---")
data3 = np.array([15, 25, 35, 45, 55])
dp_recommendation_system(data3, "max", epsilon)