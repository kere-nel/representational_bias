"""Data loading and demographics functionality."""

import json
import os
from typing import Dict, List, Tuple

def load_questions(questions_dir="/usr2/collab/kfuente1/SAE_gender/questions", occupations=None):
    """Load occupation-specific questions from text files.
    
    Args:
        questions_dir: Directory containing question files
        occupations: List of occupations to load, or None for all
        
    Returns:
        Dictionary mapping occupations to lists of questions
    """
    questions = {}
    
    if occupations is None:
        # Load all available occupations
        occupations = [f.split('.')[0] for f in os.listdir(questions_dir) 
                      if f.endswith('.txt')]
    elif occupations == 'all':
        occupations = ['surgeon', 'carpenter', 'model', 'paralegal', 'social_worker']
    elif isinstance(occupations, str):
        occupations = [occupations]
    
    for occupation in occupations:
        question_file = os.path.join(questions_dir, f"{occupation}.txt")
        if os.path.exists(question_file):
            with open(question_file, 'r') as f:
                questions[occupation] = [line.strip() for line in f if line.strip()]
        else:
            print(f"Warning: Question file not found for {occupation}")
            questions[occupation] = []
    
    return questions


def load_demographics(names_file="names_by_race_gender.json"):
    """Load demographic names data.
    
    Args:
        names_file: Path to JSON file containing names by race and gender
        
    Returns:
        Dictionary with demographic data structure
    """
    with open(names_file, 'r') as file:
        return json.load(file)


def get_demographic_prefixes(occupation):
    """Generate demographic prefixes for an occupation.
    
    Args:
        occupation: The occupation name
        
    Returns:
        Dictionary mapping experiment types to prefixes
    """
    a_occ = 'an' if occupation[0] in 'aeiou' else 'a'
    
    prefixes = {
        "baseline": f"I'm {a_occ} {occupation.replace('_', ' ')}. ",
        "none": ""
    }
    
    # Demographic categories
    pref_list = [
        "a male", "a female", "a child", "an adult", "low income", "high income",
        "never attended college", "a college graduate",
        'Hispanic', 'a Hispanic woman', 'a Hispanic man',
        'Black', 'a Black woman', 'a Black man',
        'Asian', 'an Asian woman', 'an Asian man',
        'White', 'a White woman', 'a White man'
    ]
    
    demo_list = [
        "male", "female", "child", "adult", "low-income", "high-income",
        "no_college", "college",
        'hispanic', 'hispanic_female', 'hispanic_male',
        'black', 'black_female', 'black_male',
        'asian', 'asian_female', 'asian_male',
        'white', 'white_female', 'white_male'
    ]
    
    for pref, demo in zip(pref_list, demo_list):
        prefixes[demo] = f"I'm {pref}. "
        a = 'an' if demo == 'asian' else 'a'
        if 'no_college' in demo:
            prefixes[demo] = f"I've {pref}. "
            prefixes[f"{demo}_occupation"] = f"I've {pref}, and I'm {a_occ} {occupation.replace('_', ' ')}. "
        else:
            prefixes[f"{demo}_occupation"] = f"I'm {pref}, and I'm {a_occ} {occupation.replace('_', ' ')}. "
    
    return prefixes


def create_name_based_prefixes(names_by_race_gender, occupation):
    """Create prefixes based on demographic names.
    
    Args:
        names_by_race_gender: Demographic names data
        occupation: The occupation name
        
    Returns:
        Nested dictionary structure for name-based experiments
    """
    outputs = {}
    
    for race, gender_list in names_by_race_gender.items():
        if race not in outputs:
            outputs[race] = {}
            
        for gender, names in gender_list.items():
            if gender not in outputs[race]:
                outputs[race][gender] = {}
                
            outputs[race][gender]['names'] = {
                "prefix": [],
                "outputs": [],
                "steer_outputs": [],
                "expert_scores": []
            }
            
            outputs[race][gender]['names_prof'] = {
                "prefix": [],
                "outputs": [],
                "steer_outputs": [],
                "expert_scores": []
            }
            
            for name in names:
                # Simple name prefix
                name_prefix = f"I'm {name}. "
                outputs[race][gender]['names']['prefix'].append(name_prefix)
                
                # Name + profession prefix
                prof_prefix = f"I'm {name}, and I'm a {occupation.replace('_', ' ')}. "
                outputs[race][gender]['names_prof']['prefix'].append(prof_prefix)
    
    return outputs


def partition_questions(questions, part, num_parts):
    """Partition questions for parallel processing.
    
    Args:
        questions: List of questions
        part: Current part number (1-indexed)
        num_parts: Total number of parts
        
    Returns:
        Subset of questions for this part
    """
    total_questions = len(questions)
    questions_per_part = total_questions // num_parts
    remainder = total_questions % num_parts
    
    start_idx = (part - 1) * questions_per_part + min(part - 1, remainder)
    
    if part <= remainder:
        end_idx = start_idx + questions_per_part + 1
    else:
        end_idx = start_idx + questions_per_part
    
    return questions[start_idx:end_idx]


def get_random_profession_prefixes(target_occupation, num_random_professions=3):
    """Generate random profession prefixes for baseline experiments.

    Args:
        target_occupation: The target occupation being tested
        num_random_professions: Number of random professions to use as baselines

    Returns:
        Dictionary mapping experiment types to prefixes
    """
    import random

    # Load professions from questions directory
    questions_dir = "/usr2/collab/kfuente1/SAE_gender/data/questions"
    questions_professions = [f.split('.')[0] for f in os.listdir(questions_dir)
                           if f.endswith('.txt')]

    # Additional professions to expand the pool
    additional_professions = [
        'doctor', 'lawyer', 'engineer', 'chef', 'artist', 'writer',
        'musician', 'photographer', 'architect', 'dentist', 'veterinarian',
        'pharmacist', 'librarian', 'journalist', 'scientist', 'programmer',
        'designer', 'therapist', 'consultant', 'analyst', 'researcher',
        'coordinator', 'specialist', 'barista', 'driver', 'cleaner',
        'mechanic', 'electrician', 'plumber', 'painter', 'gardener',
        'security guard'
    ]

    # Combine professions from questions directory with additional ones
    all_professions = list(set(questions_professions + additional_professions))
    
    # Define profession clusters/related groups
    profession_clusters = {
        'medical': ['doctor', 'nurse', 'surgeon', 'dentist', 'veterinarian', 'pharmacist', 'therapist', 'care_aide'],
        'tech': ['developer','engineer', 'programmer', 'scientist', 'analyst', 'researcher', 'designer'],
        'business': ['manager', 'consultant', 'accountant', 'coordinator', 'specialist'],
        'education': ['teacher', 'librarian'],
        'creative': ['artist', 'writer', 'musician', 'photographer', 'designer'],
        'legal': ['lawyer', 'paralegal'],
        'service': ['barista', 'waiter', 'cashier', 'driver', 'cleaner', 'customer_service_rep', 'receptionist', 'retail_worker'],
        'trades': ['mechanic', 'electrician', 'plumber', 'painter', 'gardener', 'carpenter', 'construction_worker'],
        'security': ['security guard'],
        'journalism': ['journalist'],
        'architecture': ['architect'],
        'food': ['chef', 'cook'],
        'care': ['social_worker',],
        'office': ['office_clerk', 'secretary', 'stocker'],
        'cleaning': ['housekeeper', 'janitor'],
        'labor': ['laborer', 'truck_driver']
    }
    
    # Find which cluster the target occupation belongs to
    target_cleaned = target_occupation.replace('_', ' ').lower()
    target_cluster = None
    for cluster_name, cluster_professions in profession_clusters.items():
        if any(target_cleaned in prof or prof in target_cleaned for prof in cluster_professions):
            target_cluster = cluster_name
            break
    
    # Filter out professions from the same cluster and exact/partial matches
    filtered_professions = []
    for profession in all_professions:
        # Skip exact matches or partial matches
        if profession.lower() in target_cleaned or target_cleaned in profession.lower():
            continue
            
        # Skip professions from the same cluster
        is_same_cluster = False
        if target_cluster:
            cluster_professions = profession_clusters[target_cluster]
            if any(profession.lower() in cluster_prof or cluster_prof in profession.lower() 
                   for cluster_prof in cluster_professions):
                is_same_cluster = True
        
        if not is_same_cluster:
            filtered_professions.append(profession)
    
    # Sample random professions
    random.seed(42)  # For reproducibility
    selected_professions = random.sample(filtered_professions, 
                                       min(num_random_professions, len(filtered_professions)))
    
    # Create prefixes
    prefixes = {}
    
    # Add baseline prefixes
    a_occ = 'an' if target_occupation[0] in 'aeiou' else 'a'
    prefixes["profession"] = f"I'm {a_occ} {target_occupation.replace('_', ' ')}. "
    
    # Add random profession prefixes
    for i, profession in enumerate(selected_professions):
        a_prof = 'an' if profession[0] in 'aeiou' else 'a'
        prefixes[f"random_{i+1}_{profession}"] = f"I'm {a_prof} {profession}. "
    
    return prefixes