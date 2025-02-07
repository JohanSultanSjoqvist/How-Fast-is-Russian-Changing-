import os
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def extract_words_from_csv(csv_path):
    # Delimiter as ';' and remove quotes
    df = pd.read_csv(csv_path, delimiter=';', quotechar='"')


    # Column names
    words = df['word_0'].tolist()    # Column name for words
    frequencies = df['hits'].tolist()  # Column name for frequencies
    return (dict(zip(words, frequencies)))

def process_folder(folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(folder_path, filename)
            all_words = extract_words_from_csv(csv_path)
    return all_words

def calculate_kl_divergence(p, q):
    """
    Calculating KL divergence between two distributions p and q,
    including all words with Laplace smoothing.
    """
    p_sum = sum(p.values())
    q_sum = sum(q.values())

    # Combining all words from both distributions
    all_words = set(p.keys()).union(set(q.keys()))
    
    kl_div = 0
    
    # Laplace Smoothing

    mini=1#min(q.values())*min(p.values())
    z_mini=mini*len(all_words)

    for word in all_words:
        

        p_prob = (p.get(word, 0)+mini ) / (p_sum + z_mini) # Smoothing
        q_prob = (q.get(word, 0)+mini ) / (q_sum + z_mini)  # Smoothing
        
        #print(f"Word: {word}, P: {p_prob}, Q: {q_prob}")  # Print probabilities

        kl_div += p_prob * np.log(p_prob ) - p_prob * np.log(q_prob)

    return kl_div

def shannon_entropy(counts):
    total_count = sum(counts.values())
    entropy = 0
    for count in counts.values():
        if count > 0:
            p_prob = count / total_count
            entropy -= p_prob * np.log(p_prob)
    return entropy



# Specify the full folder paths
folder_1_path = r'C:\Users\johan\Documents\Universitetssaker\Ryska D\Uppsats\Python\folder_1'
folder_2_path = r'C:\Users\johan\Documents\Universitetssaker\Ryska D\Uppsats\Python\folder_2'

# Process the two folders
folder_1_counts = process_folder(folder_1_path)
folder_2_counts = process_folder(folder_2_path)

kl_div_1_to_2 = calculate_kl_divergence(folder_1_counts, folder_2_counts)
kl_div_2_to_1 = calculate_kl_divergence(folder_2_counts, folder_1_counts)

# Calculate Shannon entropy
shannon_ent_1 = shannon_entropy(folder_1_counts)


print(f"KL divergence from Folder 1 to Folder 2: {kl_div_1_to_2:.10f}")
#print(f"KL divergence from Folder 2 to Folder 1: {kl_div_2_to_1:.10f}")
print(f"Shannon entropy of Folder 1: {shannon_ent_1:.10f}")

print(kl_div_1_to_2/shannon_ent_1*100)
