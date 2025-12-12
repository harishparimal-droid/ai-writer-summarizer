from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """Artificial intelligence is transforming industries worldwide. Machine learning models 
analyze vast datasets to make predictions. Natural language processing enables chatbots 
and virtual assistants. Computer vision powers self-driving cars and medical imaging.
The future holds even more exciting possibilities."""
result = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(result[0]['summary_text'])
