import json

from revllm import AnalyzeSentiment, PreprocessSentiment, VisualizeSentiment

with open('run_sentiment_inputs.txt', 'r') as infile:
    loaded_data = json.load(infile)


def choose_input(num):
    context = loaded_data[num]['context']
    ground_truth = loaded_data[num]['ground_truth']

    print(f"Context: {context}")
    print(f"Ground Truth: {ground_truth}")

    return context, ground_truth


model_name = "distilbert-base-uncased-finetuned-sst-2-english"

preprocessor = PreprocessSentiment(model_name)
analyzer = AnalyzeSentiment(model_name, preprocessor)
visualizer = VisualizeSentiment(model_name, preprocessor)

# ----------------------------------------------------------------------------------------------

context, ground_truth = choose_input(9)

preprocessor.preprocess(context, ground_truth)

analyzer.predict()

analyzer.lig_color_map()

analyzer.lig_top_k_tokens(5)

print('Visualizations')
visualizer.lc_visualize_layers()

token_to_analyze = 'but'  # also interesting> '[CLS]', behave differently from every other one

print(f"Visualizations for the specific token {token_to_analyze}")
visualizer.lc_visualize_token_boxes(token_to_analyze)
visualizer.lc_visualize_token_pdfs(token_to_analyze)
visualizer.lc_visualize_token_entropies(token_to_analyze)

print("BertViz")
visualizer.BertViz()

# ----------------------------------------------------------------------------------------------

from revllm import AnalyzeQAndA, PreprocessQAndA, VisualizeQAndA

with open('run_qanda_inputs.txt', 'r') as infile:
    loaded_data = json.load(infile)


def choose_input(num):
    question = loaded_data[num]['question']
    context = loaded_data[num]['context']
    ground_truth = loaded_data[num]['ground_truth']

    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Ground Truth: {ground_truth}")

    return question, context, ground_truth


model = 'distilbert-base-uncased-distilled-squad'

preprocessor = PreprocessQAndA(model)
analyzer = AnalyzeQAndA(model, preprocessor)
visualizer = VisualizeQAndA(model, preprocessor)

# ----------------------------------------------------------------------------------------------

question, context, ground_truth = choose_input(0)

preprocessor.preprocess(question, context, ground_truth)

analyzer.predict()

analyzer.lig_color_map()

analyzer.lig_top_k_tokens(k=5)

visualizer.lc_visualize_layers()

# choose a token from the input chosen
token_to_analyze = 'humans'
visualizer.lc_visualize_token_boxes(token_to_analyze)
visualizer.lc_visualize_token_pdfs(token_to_analyze)
visualizer.lc_visualize_token_entropies(token_to_analyze)

visualizer.BertViz()


# ----------------------------------------------------------------------------------------------

from revllm import AnalyzeMaskedLM, PreprocessMaskedLM, VisualizeMaskedLM

with open('run_maskedlm_inputs.txt', 'r') as infile:
    loaded_data = json.load(infile)


def choose_input(num):
    context = loaded_data[num]

    print(f"Context: {context}")

    return context


model_name = "distilbert-base-uncased"

preprocessor = PreprocessMaskedLM(model_name)
analyzer = AnalyzeMaskedLM(model_name, preprocessor)
visualizer = VisualizeMaskedLM(model_name, preprocessor)

# ----------------------------------------------------------------------------------------------

context = choose_input(0)
masked_substring = "France"

preprocessor.preprocess(context, masked_substring)
analyzer.predict()
analyzer.lig_color_map()
analyzer.lig_top_k_tokens()

visualizer.BertViz()
# Additional visualizer methods under construction:
