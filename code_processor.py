import os
from bs4 import BeautifulSoup
from transformers import RobertaTokenizer, RobertaModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base", trust_remote_code=True)
model.to(device)

def process_html(directory_path):
    html_pages = {}
    
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".html"):
            with open(os.path.join(directory_path, file_name), "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
            
            page_text = str(soup)
            
            embedding = get_embedding(page_text)
            html_pages[file_name] = {
                "content": page_text,
                "embedding": embedding
            }
    
    html_embeddings = [page_data["embedding"] for page_data in html_pages.values()]
    
    return html_embeddings

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()