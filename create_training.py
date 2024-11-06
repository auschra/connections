import json
import random

def process_connections_data(input_file, output_file):
    """
    Process NYT Connections game data into a format suitable for LLM fine-tuning.
    Creates instruction-response pairs with various prompt formats to improve model robustness.
    """
    
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    fine_tuning_data = []
    
    for game in data:
        date = game['date']
        answers = game['answers']
        
        # Sort answers by level to ensure consistent ordering
        answers.sort(key=lambda x: x['level'])
        
        # Create different prompt formats for the same game
        
        # Format 1: Direct question about finding groups
        words = []
        for answer in answers:
            words.extend(answer['members'])
        random.shuffle(words)  # Shuffle to prevent memorization of order
        
        instruction1 = f"Given these 16 words from the NYT Connections game: {', '.join(words)}. Find the four groups of four related words."
        response1 = "Here are the four groups:\n\n"
        for answer in answers:
            response1 += f"{answer['group']}: {', '.join(answer['members'])}\n"
        
        # Format 2: Asking about a specific group
        for answer in answers:
            instruction2 = f"In the NYT Connections game, these words appear: {', '.join(words)}. Which four words would form a group meaning '{answer['group']}'?"
            response2 = f"The words that form the group '{answer['group']}' are: {', '.join(answer['members'])}"
            
            fine_tuning_data.append({
                "instruction": instruction2,
                "response": response2
            })
        
        # Format 3: Asking for explanation of relationships
        instruction3 = f"Explain how these groups of words are related in the NYT Connections game:\n"
        for answer in answers:
            instruction3 += f"Group: {', '.join(answer['members'])}\n"
        
        response3 = "Here are the relationships:\n"
        for answer in answers:
            response3 += f"- {', '.join(answer['members'])} are related because they are all {answer['group']}\n"
        
        fine_tuning_data.append({
            "instruction": instruction1,
            "response": response1
        })
        
        fine_tuning_data.append({
            "instruction": instruction3,
            "response": response3
        })
    
    # Save the processed data
    with open(output_file, 'w') as f:
        json.dump(fine_tuning_data, f, indent=2)
    
    return len(fine_tuning_data)

# Example usage
if __name__ == "__main__":
    num_examples = process_connections_data("history.json", "fine_tuning_data.json")
    print(f"Generated {num_examples} training examples")