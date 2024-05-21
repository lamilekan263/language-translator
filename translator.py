import gradio as gr
import torch
import json
import os
# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = 'models/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d'
 
translate_text = pipeline("translation", model=model_path, torch_dtype=torch.bfloat16)

# text = 'How are you amigo'

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'language.json')

with open(file_path, 'r') as file:
    # Load the JSON data from the file
    language_data = json.load(file)


def get_Flores_code_from_language(language):
    for entry in language_data:
        if entry['Language'].lower() == language.lower():
            return entry['FLORES-200 code']
    return None


def translator(text, dest_language):
    dest_code = get_Flores_code_from_language(dest_language)
    trans = translate_text(text,src_lang='eng_Latn', tgt_lang=dest_code)
    return trans[0]['translation_text']
   


gr.close_all()

demo = gr.Interface(fn=translator, 
                    inputs=[
                        gr.Text(label="Enter text to translate", lines=6), 
                        gr.Dropdown(label='Select destination language',choices=['German','French','Hindi','Romanian'])
                        ],
                    outputs=[
                        gr.Text(label='THis is your translated text')
                        ] ,
                    title='Multi language Translator',
                    description='A Translator application')



demo.launch()




