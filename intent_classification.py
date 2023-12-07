from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_intent(sequence, labels):
    response = classifier(sequence, labels)
    return response
