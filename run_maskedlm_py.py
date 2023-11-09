from revllm.preprocess_distilbert import PreprocessMaskedLM
from revllm.analyze_distilbert import AnalyzeMaskedLM
# Under construction:
# from revllm.visualize_distilbert import VisualizeMaskedLM 

model_name = "distilbert-base-uncased"

preprocessor = PreprocessMaskedLM(model_name)
analyzer = AnalyzeMaskedLM(model_name, preprocessor)
# Under construction:
# visualizer = VisualizeMaskedLM(model_name, preprocessor)

context = "The capital of France, Paris, contains the Eiffel Tower."
masked_substring = "Tower"

preprocessor(context,masked_substring)
analyzer.predict()
analyzer.lig_color_map()
analyzer.lig_top_k_tokens()