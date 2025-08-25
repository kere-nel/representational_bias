import pandas as pd
import re
import textstat
def clean_demographic_reference(text):
    if not isinstance(text, str):
        return text
    # Common demographic introduction patterns to remove
    patterns = [
        r"^\s*As a [^,.:;]+[,.:;]?\s*",       # e.g., "As a woman," "As a gay man:"
        r"^\s*Being (a|an) [^,.:;]+[,.:;]?\s*", # e.g., "Being a Muslim,"
        r"^\s*I'm (a|an) [^,.:;]+[,.:;]?\s*",   # e.g., "I'm a Black developer,"
        r"^\s*I am (a|an) [^,.:;]+[,.:;]?\s*",  # e.g., "I am an Asian man."
        r"^\s*As [^,.:;]+[,.:;]?\s*",           # e.g., "As someone who is disabled,"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()
    
def readability_score(text): 
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
        return None
    cleaned = clean_demographic_reference(text)
    return (textstat.dale_chall_readability_score(cleaned) + textstat.flesch_kincaid_grade(cleaned) + flesch_reading_ease_to_grade(textstat.flesch_reading_ease(cleaned)))/3
    
def flesch_reading_ease_to_grade(score):
    if score >= 90: return 5
    elif score >= 80: return 6
    elif score >= 70: return 7
    elif score >= 60: return 8
    elif score >= 50: return 10
    elif score >= 30: return 12
    else: return 16
        

def get_readability(df):
    # Apply to each column and compute the average
# Step 1: Compute mean readability score per column per question
    per_question_means = df.drop(columns=['question']).applymap(readability_score)
    per_question_means['question'] = df['question']
    grouped_means = per_question_means.groupby('question').mean()
    
    # Convert to DataFrame for display
    #result_df = pd.DataFrame.from_dict(column_averages, orient='index', columns=['Average Dale-Chall Score'])
    #result_df.index.name = 'Column'
    #return result_df.round(2)
    return grouped_means

def remove_context(text):
    # Only split if there's at least one period followed by a space
    if '. ' in text:
        parts = text.split('. ')
        return '. '.join(parts[1:]).strip()
    return text.strip()
def remove_context_post(text):
    # Only split if there's at least one period followed by a space
    if 'For reference,' in text:
        parts = text.split(' For reference')
        return parts[0]
    return text.strip()