import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

def compare_audio_files(features1, features2):

    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
    return similarity[0][0]



def detect_person(input_rec_features):
    res = {}
    # Specify the file name
    file_name = 'dictionary.pkl'

    # Get the current working directory
    current_directory = os.getcwd()

    # Create the file path by joining the current directory and the file name
    file_path = "pickles/dictionary.pkl"

    # Check if the file is empty
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print("Error: The pickle file is empty.")
    else:
        # Load the dictionary from the pickle file
        with open(file_path, 'rb') as pickle_file:
            try:
                loaded_dict = pickle.load(pickle_file)
                for key, value in loaded_dict.items():
                    res[key] = compare_audio_files(value, input_rec_features)
            except EOFError:
                print("Error: The pickle file does not contain valid data.")

    return res

