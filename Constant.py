class Constant:
    CLASSES = ["class-0", "class-1", "class-2", "class-3"]
    
    DATASET_DIR = "dataset"

    DATASET_DELIMETER = ","
    
    NEWLINE = "\n"

    BREAKPOINT = "======" * 10

# Some helper functions
def view_dataset(feature_dataset, label_dataset):
    for i in range(len(feature_dataset)):
        print(f'features - {feature_dataset[i]} label - {label_dataset[i]}')