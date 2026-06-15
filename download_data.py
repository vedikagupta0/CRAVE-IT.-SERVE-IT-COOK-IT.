import kagglehub
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

path = kagglehub.dataset_download("seungyeonhan1/recipe-dataset-with-images-tags-and-ratings")

print("Path to dataset files:", path)