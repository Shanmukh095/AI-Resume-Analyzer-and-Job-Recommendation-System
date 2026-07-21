
from matcher import JobMatcher


def evaluate():

    print("Initializing MPNet Job Matcher...\n")

    matcher = JobMatcher()


    resume = """

    Experienced Python developer with expertise in
    Machine Learning,
    Deep Learning,
    Natural Language Processing,
    SQL,
    TensorFlow,
    Data Analysis,
    REST APIs,
    Flask,
    Django.

    """


    recommendations = matcher.recommend(
        resume_text=resume,
        top_k=5
    )


    print("\n========== TOP 5 RECOMMENDED JOBS ==========\n")


    for i, job in enumerate(recommendations, start=1):

        print(f"{i}. Job Title   : {job['job_title']}")
        print(f"   Industry   : {job['industry']}")
        print(f"   Match Score: {job['match_score']}%")
        print("-" * 45)


if __name__ == "__main__":

    evaluate()
