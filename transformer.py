from transformers import pipeline

classifier = pipeline("zero-shot-classification") # the models are installed to this directory: ~/.cache/huggingface/hub
response = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(response)
