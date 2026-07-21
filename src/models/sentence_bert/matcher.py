
import torch
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


MODEL_NAME = "all-MiniLM-L6-v2"


class JobMatcher:


    def __init__(self):

        print("Loading Sentence-BERT...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )


        self.job_embeddings = torch.load(
            "saved_models/sentence_bert/job_embeddings.pt"
        )


        self.jobs = pd.read_csv(
            "saved_models/sentence_bert/job_metadata.csv"
        )


        print("Matcher ready")



    def recommend(
        self,
        resume_text,
        top_k=5
    ):


        resume_embedding = self.model.encode(
            resume_text,
            convert_to_tensor=True
        )


        scores = cos_sim(
            resume_embedding,
            self.job_embeddings
        )[0]


        top_results = torch.topk(
            scores,
            k=top_k
        )


        recommendations = []


        for score, index in zip(
            top_results.values,
            top_results.indices
        ):


            job = self.jobs.iloc[index.item()]


            recommendations.append({

                "job_title":
                    job["title"],

                "industry":
                    job["industry_name"],

                "match_score":
                    round(
                        score.item()*100,
                        2
                    )

            })


        return recommendations



if __name__ == "__main__":


    matcher = JobMatcher()


    resume = """
    Python developer with experience
    in machine learning, deep learning,
    NLP, SQL and data analysis
    """


    results = matcher.recommend(
        resume,
        top_k=5
    )


    print("\nRecommended Jobs\n")


    for job in results:

        print(job)
