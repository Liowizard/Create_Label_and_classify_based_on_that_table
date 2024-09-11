from collections import Counter

import pandas as pd
import spacy
from bson import ObjectId
from fuzzywuzzy import fuzz
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/test")
db = client["test"]
collection = db["digisurveyresponses"]

# MongoDB query filter
filter_query = {
    "surveyId": ObjectId("66d696a7bc7dff6adc1d21da"),
    "sectionNo": "Section B",
    "questionNo": "question4",
}

# Retrieve data from MongoDB
documents = list(collection.find(filter_query, {"_id": 1, "answer": 1}))

reviews = [doc["answer"] for doc in documents]
doc_ids = [doc["_id"] for doc in documents]

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])


def preprocess_texts(texts):
    processed_reviews = []
    for doc in nlp.pipe(texts, batch_size=50):
        tokens = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]
        processed_reviews.append(" ".join(tokens))
    return processed_reviews


processed_reviews = preprocess_texts(reviews)

df = pd.DataFrame(
    {"_id": doc_ids, "review": reviews, "processed_review": processed_reviews}
)

vectorizer = TfidfVectorizer(max_features=20)
tfidf_matrix = vectorizer.fit_transform(df["processed_review"])
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray().sum(axis=0)
keyword_scores = dict(zip(feature_names, tfidf_scores))

top_keywords = [keyword for keyword, score in Counter(keyword_scores).most_common(50)]

keywords_to_exclude = {}

top_filtered_keywords = [
    keyword for keyword in top_keywords if keyword not in keywords_to_exclude
]
top_3_keywords = top_filtered_keywords[:3]

print("Top 3 Keywords:", top_3_keywords)


def keyword_match(review, keyword, threshold=80):
    ratio = fuzz.partial_ratio(review, keyword)
    return ratio >= threshold


for keyword in top_3_keywords:
    df["related_result"] = df["processed_review"].apply(
        lambda review: keyword_match(review, keyword)
    )
    related_reviews = df[df["related_result"] == True]
    for _, row in related_reviews.iterrows():
        collection.update_one({"_id": row["_id"]}, {"$addToSet": {"category": keyword}})
