import json
import random

input_file = 'datasets/history.json'  
output_file = 'datasets/ft_1.json'

# load json
with open(input_file, 'r') as f:
    data = json.load(f)

# dataset augmentation factor
augmentation_factor = 2

# new dataset
expanded_dataset = []

# create new entries
def create_shuffled_entry(entry):

    # words
    groups = entry['answers']
    words = [word for group in groups for word in group['members']]
    words = [word.lower() for word in words]
    words = ", ".join(words)

    # theme
    theme = [group['group'] for group in entry['answers']]

    response_parts = []
    
    response = "\n".join(response_parts)

    for group in word_groups:
        # Shuffle the words within each group
        random.shuffle(group)
        response_parts.append(f"({' '.join(group)})")

    response = f"{' '.join(response_parts)}"

    return {
        "instruction": instruction,
        "words": words,
        "response": response
}

# Expand the dataset by generating multiple shuffled versions for each entry
for entry in data[:4]:
    for _ in range(augmentation_factor):
        shuffled_entry = create_shuffled_entry(entry)
        expanded_dataset.append(shuffled_entry)

# Shuffle the entire expanded dataset
random.shuffle(expanded_dataset)

# Save the expanded dataset
with open(output_file, 'w') as f:
    json.dump(expanded_dataset, f, indent=4)

print(f"Expanded finetuning dataset saved to {output_file}")