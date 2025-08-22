
import csv
import copy
import os
from datetime import datetime

# CSV file headers
TRAIN_FILE_HEADER = ["local_model", "flr", "average_loss", "accuracy", "correct_data", "total_data"]
TEST_FILE_HEADER = ["model", "flr", "avg_clean_loss", "avg_poison_loss", "avg_clean_acc", "avg_poison_acc"]
TRIGGER_TEST_FILE_HEADER = ["model", "flr", "average_loss", "accuracy", "correct_data", "total_data"]
REALTIME_TEST_HEADER = ["timestamp", "test_type", "fl_round", "model_identifier", "clean_loss", "poison_loss", "clean_accuracy", "poison_accuracy", "test_eps", "dataset", "defense_method", "attack_method"]

# Global result storage
train_result = []
test_result = []
poison_test_result = []
poison_trigger_test_result = []
weight_result = []
scale_result = []
scale_temp_one_row = []

# Real-time test tracking
realtime_test_results = []
realtime_csv_file = None
realtime_csv_writer = None

def initialize_realtime_csv_writer(output_dir="results", filename_prefix="realtime_test_results"):
    """Initialize the real-time CSV writer for test results.
    
    Args:
        output_dir (str): Directory to save CSV files
        filename_prefix (str): Prefix for the CSV filename
    """
    global realtime_csv_file, realtime_csv_writer
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{filename_prefix}_{timestamp}.csv"
    
    # Open CSV file and create writer
    realtime_csv_file = open(filename, "w", newline='')
    realtime_csv_writer = csv.writer(realtime_csv_file)
    realtime_csv_writer.writerow(REALTIME_TEST_HEADER)
    
    print(f"Initialized real-time CSV writer: {filename}")
    return filename

def write_realtime_test_result(test_type, fl_round, model_identifier, clean_loss, poison_loss, 
                              clean_accuracy, poison_accuracy, test_eps, dataset, defense_method, 
                              attack_method):
    """Write a test result immediately to the CSV file.
    
    Args:
        test_type (str): Type of test (e.g., 'Internal', 'External', 'Baseline')
        fl_round (int): Federated learning round number
        model_identifier (str): Identifier for the model being tested
        clean_loss (float): Clean data loss
        poison_loss (float): Poisoned data loss
        clean_accuracy (float): Clean data accuracy
        poison_accuracy (float): Poisoned data accuracy
        test_eps (float): Test epsilon value
        dataset (str): Dataset name
        defense_method (str): Defense method used
        attack_method (str): Attack method used
    """
    global realtime_csv_writer
    
    if realtime_csv_writer is None:
        # Initialize if not already done
        initialize_realtime_csv_writer()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    row = [
        timestamp,
        test_type,
        fl_round,
        model_identifier,
        f"{clean_loss:.6f}",
        f"{poison_loss:.6f}",
        f"{clean_accuracy:.6f}",
        f"{poison_accuracy:.6f}",
        f"{test_eps:.6f}",
        dataset,
        defense_method,
        attack_method
    ]
    
    realtime_csv_writer.writerow(row)
    realtime_csv_file.flush()  # Ensure data is written immediately
    
    # Also store in memory for later use
    realtime_test_results.append(row)

def close_realtime_csv_writer():
    """Close the real-time CSV writer."""
    global realtime_csv_file
    if realtime_csv_file is not None:
        realtime_csv_file.close()
        realtime_csv_file = None
        print("Closed real-time CSV writer")

def save_result_csv(epoch, is_poison, folder_path):
    """Save training and testing results to CSV files.
    
    Args:
        epoch (int): Current epoch number
        is_poison (bool): Whether this is a poison test
        folder_path (str): Path to save CSV files
    """
    # Save training results
    train_csv_file = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csv_file)
    train_writer.writerow(TRAIN_FILE_HEADER)
    train_writer.writerows(train_result)
    train_csv_file.close()

    # Save testing results
    test_csv_file = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csv_file)
    test_writer.writerow(TEST_FILE_HEADER)
    test_writer.writerows(test_result)
    test_csv_file.close()

def add_weight_result(name, weight, alpha):
    """
    
    Args:
        name (str): Name identifier
        weight (float): Weight value
        alpha (float): Alpha parameter
    """
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)

def add_scale_result(scale, alpha):
    """
    
    Args:
        scale (float): Scale value
        alpha (float): Alpha parameter
    """
    scale_result.append(scale)
    scale_result.append(alpha)

def add_scale_temp_one_row(scale, alpha):
    """
    
    Args:
        scale (float): Scale value
        alpha (float): Alpha parameter
    """
    scale_temp_one_row.append(scale)
    scale_temp_one_row.append(alpha)

def save_scale_result_csv(folder_path):
    """Save scale results to CSV file.
    
    Args:
        folder_path (str): Path to save CSV file
    """
    scale_csv_file = open(f'{folder_path}/scale_result.csv', "w")
    scale_writer = csv.writer(scale_csv_file)
    scale_writer.writerow(["scale", "alpha"])
    scale_writer.writerows([scale_result[i:i+2] for i in range(0, len(scale_result), 2)])
    scale_csv_file.close()

def save_scale_temp_one_row_csv(folder_path):
    """Save scale temp one row results to CSV file.
    
    Args:
        folder_path (str): Path to save CSV file
    """
    scale_temp_csv_file = open(f'{folder_path}/scale_temp_one_row.csv', "w")
    scale_temp_writer = csv.writer(scale_temp_csv_file)
    scale_temp_writer.writerow(["scale", "alpha"])
    scale_temp_writer.writerows([scale_temp_one_row[i:i+2] for i in range(0, len(scale_temp_one_row), 2)])
    scale_temp_csv_file.close()

def save_weight_result_csv(folder_path):
    """Save weight results to CSV file.
    
    Args:
        folder_path (str): Path to save CSV file
    """
    weight_csv_file = open(f'{folder_path}/weight_result.csv', "w")
    weight_writer = csv.writer(weight_csv_file)
    weight_writer.writerow(["name", "weight", "alpha"])
    weight_writer.writerows([weight_result[i:i+3] for i in range(0, len(weight_result), 3)])
    weight_csv_file.close()

def get_realtime_test_summary():
    """Get a summary of all real-time test results.
    
    Returns:
        dict: Summary statistics of test results
    """
    if not realtime_test_results:
        return {}
    
    # Extract numeric columns (skip timestamp, test_type, model_identifier, dataset, defense_method, attack_method)
    clean_accuracies = [float(row[6]) for row in realtime_test_results[1:]]  # Skip header
    poison_accuracies = [float(row[7]) for row in realtime_test_results[1:]]
    
    summary = {
        'total_tests': len(realtime_test_results) - 1,  # Exclude header
        'avg_clean_accuracy': sum(clean_accuracies) / len(clean_accuracies) if clean_accuracies else 0,
        'avg_poison_accuracy': sum(poison_accuracies) / len(poison_accuracies) if poison_accuracies else 0,
        'max_clean_accuracy': max(clean_accuracies) if clean_accuracies else 0,
        'max_poison_accuracy': max(poison_accuracies) if poison_accuracies else 0,
        'min_clean_accuracy': min(clean_accuracies) if clean_accuracies else 0,
        'min_poison_accuracy': min(poison_accuracies) if poison_accuracies else 0,
    }
    
    return summary


