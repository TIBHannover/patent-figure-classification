import re
import random

from model_utils import generate, postprocess

random.seed(1337)

def create_multiple_choice_question(template, options_list):
    """
        Create a multiple question from a template

        Args:
            template: multiple question template to use
            options_list: classification labels to use in the multiple-choice question
        
        Returns:
            Multiple-choice question
    """

    question = template["question"]

    options_text = [f"({i+1}) {options_list[i]}" for i in range(min(len(options_list), len(options_list)+1))]

    question = question.replace(" {options_text}", " ".join(options_text))

    return question

def classify(model, loader, template, options, t, level):
    """
        Classify the figures using multiple-choice tournament classification approach

        Args:
            model: LVLM to use for classification
            loader: figures to classify
            template: multiple-choice classification template to use for classification
            options: all possible classification labels
            t: number of options to use
            level: round of tournament

        Returns:
            Winner of the tournament classification approach
    """
    level += 1
    print(f"Level: {level}")
    
    last_key = list(options.keys())[-1]
    
    if len(options[last_key]["concepts"]) == 1: # last sample has 1 option
        return options
    else:
        if len(options[last_key]["concepts"]) >= t:

            for id, option_list in options.items():

                option_list = option_list["concepts"]

                if isinstance(option_list, str):
                    option_list = list(option_list)
                
                random.shuffle(option_list)

                options[id]["concepts"] = [
                    option_list[i:i+t] for i in range(0, len(option_list), t)]
        else:
            for id, option_list in options.items():
                
                options_list = options[id]["concepts"]
                random.shuffle(options_list)

                options[id]["concepts"] = [options_list]

        for i in range(len(options[last_key]["concepts"])):
            
            for batch in loader:

                ids = batch[0]
                figures = batch[1]
                reference_answers = batch[2]

                questions = [
                    create_multiple_choice_question(template, options_list=options[id]["concepts"][i])
                        for id in ids
                ]

                outputs, _ = generate(model, figures, questions)

                for id, answer, question, reference_answer in zip(ids, outputs, questions, reference_answers):
                    try:
                        intermediate_answer = options[id]["concepts"][i][postprocess(answer)-1]
                    except IndexError:
                        intermediate_answer = "None"

                    options[id].setdefault("option_list", {})
                    options[id]["option_list"].setdefault(level, [])
                    options[id]["option_list"][level].append(question.split("Options:")[-1])

                    options[id]["concepts"][i] = intermediate_answer
                    options[id]["generated_answer"] = answer
                    options[id]["reference_answer"] = reference_answer

        return classify(model, loader, template, options, t, level)

def get_top_k(answers, k):
    """Select the top-k classification label"""
    pattern = r'\(\d+\)\s*((?:\w+(?:\s+\w+)*)+)'

    for id, sample in answers.items():
        options_list_dict = sample["option_list"]
        candidates = []
        for i in range(k, 0):
            options_list_last_key = list(options_list_dict.keys())[k]
            for candidates_string in options_list_dict[options_list_last_key]:
                candidates_string = candidates_string.replace(".","")
                candidates.extend(re.findall(pattern, candidates_string))
            answers[id]["top_k"] = candidates

    return answers

def compute_accuracy(answers):
    """Compute accuracy by comparing reference answer to selected answer"""
    acc = 0.0

    for id, sample in answers.items():
        reference_answer = sample["reference_answer"].lower()
        prediction = sample['concepts'][0].lower()

        acc += 1.0 if prediction==reference_answer else 0.0
    
    accuracy = (acc / len(answers)) * 100
    print(f"{accuracy:.2f}%")