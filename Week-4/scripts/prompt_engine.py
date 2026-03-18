import pandas as pd

class PromptEngine:
    """
    Manages prompt templates and strategies for Day 2 Experiments.
    """
    def __init__(self, shot_bank=None):
        """
        shot_bank: A DataFrame containing examples for Few-Shot prompting.
        """
        self.shot_bank = shot_bank

    def generate_prompt(self, text, strategy):
        # Truncate input text to avoid context limits
        text = text[:1000]
        
        # 1. Zero-Shot Basic
        if strategy == "Zero-Shot Basic":
            return f"""Classify the sentiment of this movie review as "Positive" or "Negative".
Review: "{text}"
Sentiment:"""

        # 2. Zero-Shot Persona
        elif strategy == "Zero-Shot Persona":
            return f"""You are an expert film critic. Analyze the following review.
If the reviewer liked the movie, say "Positive". If they disliked it, say "Negative".
Review: "{text}"
Sentiment:"""

        # 3. Few-Shot (1-shot)
        elif strategy == "Few-Shot (1-shot)":
            if self.shot_bank is None or len(self.shot_bank) < 1:
                return "Error: No shot bank provided."
                
            ex = self.shot_bank.iloc[0]
            # Handle potential labeling differences (0/1 vs Pos/Neg)
            lbl = "Positive" if ex['label'] == 1 else "Negative"
            
            return f"""Task: Classify Movie Reviews.
Example:
Review: "{ex['text'][:200]}..."
Sentiment: {lbl}

Review: "{text}"
Sentiment:"""

        # 4. Few-Shot (3-shot)
        elif strategy == "Few-Shot (3-shot)":
            if self.shot_bank is None or len(self.shot_bank) < 3:
                return "Error: Shot bank too small."
            
            examples = ""
            for i in range(3):
                ex = self.shot_bank.iloc[i]
                lbl = "Positive" if ex['label'] == 1 else "Negative"
                examples += f'Review: "{ex["text"][:150]}..."\nSentiment: {lbl}\n\n'
                
            return f"""Task: Classify Movie Reviews.
{examples}
Review: "{text}"
Sentiment:"""

        # 5. Chain of Thought (CoT)
        elif strategy == "Chain of Thought":
            return f"""Analyze the movie review. 
Step 1: Identify emotional keywords.
Step 2: Decide if the tone is Positive or Negative.
Format:
Reasoning: [Reasoning]
Sentiment: [Label]

Review: "{text}"
"""
        return text