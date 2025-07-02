<div align="center">

# 💊 VitalCare GPT

**AI-Powered Multi-Role Medical Assistant using Hugging Face Transformers & Streamlit**

![Last Commit](https://img.shields.io/github/last-commit/Muhammad-Ahmed-Rayyan/VitalCare-Custom-GPT)
![Python](https://img.shields.io/badge/Python-100%25-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface)
![Torch](https://img.shields.io/badge/torch-enabled-orange?logo=pytorch)
![languages](https://img.shields.io/github/languages/count/Muhammad-Ahmed-Rayyan/VitalCare-Custom-GPT)

</div>

---

## 🧠 Project Summary

**VitalCare GPT** is a domain-specific multi-role chatbot platform that utilizes fine-tuned language models hosted on Hugging Face to assist users in three medical domains:

- 👨‍⚕️ **Doctor:** Diagnosis & clinical guidance  
- 🥗 **Nutritionist:** Meal plans & dietary advice  
- 💊 **Pharmacist:** Medication and dosage information

Built with **Streamlit** for an intuitive user interface and trained via the **Hugging Face Transformers ecosystem**, this project demonstrates how multiple LLMs can be orchestrated in a single application based on user context.

**🔗[Try it Live on Streamlit](https://vitalcare-custom-gpt.streamlit.app)**

---

## 🚀 Features

- 🧬 **Fine-Tuned Medical Models**  
  Multiple transformer-based language models hosted on Hugging Face:
  - `PastelMed`
  - `LynxMed`
  - `NeuraMed`
  - `SkyeMed`
  - `ClixMed`

- 📋 **Role-Specific Spaces**  
  Each space uses domain-specific prompts and keyword validation for accuracy:
  - Doctor, Nutritionist, Pharmacist

- 💬 **Chat-Like GUI in Streamlit**  
  Real-time text generation using `pipeline()` with history tracking for each space.

- 🧠 **Custom Model Training & Upload**  
  Notebook provided for fine-tuning custom transformers and uploading to Hugging Face.

---

## 🗃️ Project Structure

```bash
├── app.py                         # Streamlit app with role-based LLM chat
├── requirements.txt              # Project dependencies
```

---

## 🔧 Setup & Installation

> Make sure Python 3.8+ is installed.

```bash
# Clone the repo
git clone https://github.com/Muhammad-Ahmed-Rayyan/VitalCare-GPT.git
cd VitalCare-GPT

# Install required libraries
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📊 Model Fine-Tuning

All training is performed using Hugging Face's `transformers` and `datasets` libraries in the included notebook.

### `Model_Training_&_Upload.ipynb` Includes:
- Loading and inspecting datasets
- Using tokenizer & model
- Preprocessing and training
- Saving and uploading models to Hugging Face Hub

---

## ☁️ Hosting on Hugging Face

Once the model is trained and saved, it can be uploaded to your Hugging Face model hub using:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path = "./fine_tuned_model",     # Folder where model is saved
    repo_id = "username/model-name",        # Your Hugging Face repo name
    repo_type = "model"
)

