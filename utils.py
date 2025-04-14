import os
import re
import sys

from matplotlib import pyplot as plt

def remove_joint_questions(train_questions, test_questions, test_answers):
    # Find indices of questions in test_questions that also appear in train_questions
    indices_to_remove = [i for i, q in enumerate(test_questions) if q in train_questions]

    print(f"There are {len(indices_to_remove)} questions in both dataset!")
    
    # Create new lists without the joint elements
    filtered_test_questions = [q for i, q in enumerate(test_questions) if i not in indices_to_remove]
    filtered_test_answers = [a for i, a in enumerate(test_answers) if i not in indices_to_remove]
    
    return filtered_test_questions, filtered_test_answers

def extract_rag_answer(text):
    # Find all "Answer:" blocks
    matches = list(re.finditer(r"Answer:\s*(.+?)(?=\n(?:Question:|Answer:)|\Z)", text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None

def extract_inst_only_model_answer(text):
    match = re.search(r"### Response:\s*(.+)", text, re.DOTALL)
    return match.group(1).strip() if match else None