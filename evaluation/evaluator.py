from evaluation.sample_questions import QUESTIONS


def run_evaluation():

    print("\nEvaluation Dataset\n")

    for item in QUESTIONS:

        print(
            f"Question: "
            f"{item['question']}"
        )