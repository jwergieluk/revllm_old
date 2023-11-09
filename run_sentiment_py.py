from revllm.preprocess_distilbert import PreprocessSentiment
from revllm.analyze_distilbert import AnalyzeSentiment
from revllm.visualize_distilbert import VisualizeSentiment

context = "The movie had breathtaking visuals, but the storyline left a lot to be desired."
ground_truth = "negative"

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

preprocessor = PreprocessSentiment(model_name)
analyzer = AnalyzeSentiment(model_name, preprocessor)
visualizer = VisualizeSentiment(model_name, preprocessor)

preprocessor(context, ground_truth)

print(preprocessor.input_ids)

analyzer.predict()

analyzer.lig_color_map()

analyzer.lig_top_k_tokens(5)

# start here - also, this will move to visualizer
# analyzer.BertViz()

visualizer.lc_visualize_layers()

token_to_analyze = 'but' #also interesting> '[CLS]', behave differently from every other one
visualizer.lc_visualize_token_boxes(token_to_analyze)
visualizer.lc_visualize_token_pdfs(token_to_analyze)
visualizer.lc_visualize_token_entropies(token_to_analyze)