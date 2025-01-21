from tqdm import tqdm
from model_utils import generate

def create_binary_question(template, concept):
    """
        Create a binary question from a template

        Args:
            template: binary question template to use
            concept: classification concept the binary questio is about
        
        Returns:
            Binary question
    """

    question = template["question"]
    question_aspect = template["aspect"]
    question = question.replace("{"+f"{question_aspect}"+"}", concept)

    return question

def classify(model, loader, template, options):
    """
        Classify the figures using binary classification approach

        Args:
            model: LVLM to use for classification
            loader: figures to classify
            template: binary classification template to use for classification
            options: all possible classification labels

        Returns:
            List of answers for each figure for each classification label
    """
    answers = {}

    last_key = list(options.keys())[-1]
    num_of_concepts = len(options[last_key]["concepts"])

    for i in tqdm(range(num_of_concepts)):

        for batch in loader:

            ids = batch[0]
            figures = batch[1]
            reference_answers = batch[2]

            questions = [create_binary_question(template, options[id]["concepts"][i]) for id in ids]
            outputs, scores = generate(model, figures, questions)

            for id, answer, score, reference_answer in zip(ids, outputs, scores, reference_answers):
                answers.setdefault(id, {
                    "reference_answer": reference_answer,
                    "generated_answer": answer,
                    "responses": []
                })

                if answer.lower() in ["y", "yes"]:
                    answers[id]["responses"].append(
                        (options[id]["concepts"][i], score)
                    )

    return answers

def get_top_k(answers, k):
    """Select the top-k classification labels based on score"""
    for id, sample in answers.items():
        responses = sample["responses"]

        if responses:
            sorted_answers = list(sorted(responses, key=lambda x: x[1], reverse=True))
            top_k = [answer for answer, score in sorted_answers]
            answers[id]["top_k"] = top_k[:k]
        else:
            answers[id]["top_k"] = []

    return answers

def compute_accuracy(answers):
    """Compute accuracy by comparing reference answer to selected answer"""
    acc = 0.0

    for id, sample in answers.items():
        reference_answer = sample["reference_answer"]
        responses = sample["responses"]

        if responses:
            sorted_answers = list(sorted(responses, key=lambda x: x[1], reverse=True))
            selected_answer = sorted_answers[0][0]
            answers[id]["selected_answer"] = selected_answer
            if selected_answer and selected_answer == reference_answer:
                acc += 1.0
    
    accuracy = (acc / len(answers)) * 100
    print()
    print(f"{accuracy:.2f}%")

    return answers