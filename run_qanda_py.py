from revllm.preprocess_distilbert import PreprocessQAndA
from revllm.analyze_distilbert import AnalyzeQAndA
from revllm.visualize_distilbert import VisualizeQAndA

question = "What is important to us?"
context = "It is important to us to include, empower and support humans of all kinds."
ground_truth = "to include, empower and support humans of all kinds"

model = 'distilbert-base-uncased-distilled-squad'

preprocessor = PreprocessQAndA(model)
analyzer = AnalyzeQAndA(model,preprocessor)
visualizer = VisualizeQAndA(model,preprocessor)

preprocessor(question, context, ground_truth)

analyzer.predict()

analyzer.lig_color_map()

analyzer.lig_top_k_tokens(k=5)

visualizer.lc_visualize_layers()

token_to_analyze = 'humans'
visualizer.lc_visualize_token_boxes(token_to_analyze)
visualizer.lc_visualize_token_pdfs(token_to_analyze)
visualizer.lc_visualize_token_entropies(token_to_analyze)