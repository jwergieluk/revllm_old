import json

from revllm import AnalyzeMaskedLM
from revllm import PreprocessMaskedLM
from revllm import VisualizeMaskedLM

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
