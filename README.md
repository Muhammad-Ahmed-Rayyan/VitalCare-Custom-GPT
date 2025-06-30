<div align="center">

# 💊 VitalCare GPT

**AI-Powered Multi-Role Medical Assistant using Hugging Face Transformers & Streamlit**

![Last Commit](https://img.shields.io/github/last-commit/Muhammad-Ahmed-Rayyan/VitalCare-GPT)
![Python](https://img.shields.io/badge/Python-100%25-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface)
![Torch](https://img.shields.io/badge/torch-enabled-orange?logo=pytorch)

</div>

---

## 🧠 Project Summary

**VitalCare GPT** is a domain-specific multi-role chatbot platform that utilizes fine-tuned language models hosted on Hugging Face to assist users in three medical domains:

- 👨‍⚕️ **Doctor:** Diagnosis & clinical guidance  
- 🥗 **Nutritionist:** Meal plans & dietary advice  
- 💊 **Pharmacist:** Medication and dosage information

Built with **Streamlit** for an intuitive user interface and trained via the **Hugging Face Transformers ecosystem**, this project demonstrates how multiple LLMs can be orchestrated in a single application based on user context.

---

## 🚀 Features

- 🧬 **Fine-Tuned Medical Models**  
  Multiple transformer-based language models (BioGPT variations) hosted on Hugging Face:
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
├── Model_Training_&_Upload.ipynb # Fine-tuning + Upload to Hugging Face
