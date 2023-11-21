import json

from revllm import AnalyzeQAndA
from revllm import PreprocessQAndA
from revllm import VisualizeQAndA

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
