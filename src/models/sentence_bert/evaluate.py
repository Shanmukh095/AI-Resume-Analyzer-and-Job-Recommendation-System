
from matcher import JobMatcher



def evaluate():

    matcher = JobMatcher()


    resume = """

    Experienced software developer
    with Python, Java, SQL,
    machine learning and NLP skills.

    Worked on data analysis,
    deep learning and backend systems.

    """


    results = matcher.recommend(
        resume,
        top_k=5
    )


    print("\nTop Recommended Jobs\n")


    for i, job in enumerate(results, 1):

        print(
            f"{i}. {job}"
        )



if __name__ == "__main__":

    evaluate()
