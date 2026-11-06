import json
import gzip
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def compress_json(input_filepath: str, output_filepath: str):
    if not os.path.exists(input_filepath):
        logging.error(f"Fichier introuvable : {input_filepath}")
        return
        
    logging.info(f"Lecture de {input_filepath} ({os.path.getsize(input_filepath) / (1024*1024):.2f} MB)...")
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    logging.info(f"Compression en cours vers {output_filepath}...")
    
    # On compresse sans indentation pour maximiser le gain de place
    with gzip.open(output_filepath, 'wt', encoding='utf-8') as gz_file:
        json.dump(data, gz_file, separators=(',', ':'))
        
    final_size = os.path.getsize(output_filepath) / (1024*1024)
    logging.info(f"✅ Compression terminée ! Nouvelle taille : {final_size:.2f} MB")
    logging.info(f"Tu peux maintenant uploader {output_filepath} sur GitHub.")

if __name__ == "__main__":
    compress_json("xgboost_runes_model_V3.json", "xgboost_runes_model_V3.json.gz")