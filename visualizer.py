import numpy as np

def detect_data_type(data):
    
    data = np.array(data)

    # If only 0 and 1 then binary
    unique_values = np.unique(data)

    if set(unique_values).issubset({0, 1}):
        return "binary"
    
    # If numeric values
    elif np.issubdtype(data.dtype, np.number):
        return "numerical"
    
    else:
        return "categorical"


def detect_query_type(query):
    
    query = query.lower()

    if query in ["mean", "average", "sum", "count"]:
        return "numerical"
    
    elif query in ["survey", "binary"]:
        return "binary"
    
    elif query in ["argmax", "max", "selection"]:
        return "selection"
    
    else:
        return "unknown"


def get_applicable_mechanisms(data_type, query_type):
   

    if query_type == "numerical":
        return ["laplace", "gaussian"]
    
    elif query_type == "binary":
        return ["randomized_response"]
    
    elif query_type == "selection":
        return ["exponential"]
    
    else:
        return []