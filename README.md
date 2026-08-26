# Data-Efficient Fine-Tuning of Vision-Language Models for Diagnosis of Alzheimer’s Disease

## Overview
DEFT is a data-efficient fine-tuning framework for 3D medical vision-language models (Med-VLMs), designed for Alzheimer's disease diagnosis.  
It leverages structured metadata to generate synthetic medical reports and introduces an auxiliary MMSE token for cognitive score prediction.  
Our approach achieves strong performance on ADNI, with zero-shot generalization to OASIS-2 and AIBL datasets.

## Features
- 3D Vision Transformer (M3D) for MRI feature extraction
- Learnable prompts and MMSE token tuning
- Cross-attention between image and text modalities
- CLIP-style contrastive loss
- MMSE cognitive score prediction from MRI features
- Automatic medical report generation from MRI-derived biomarkers (Hippocampus, Ventricles, Whole Brain, Entorhinal volumes) and clinical labels

## Dataset
This project uses publicly available Alzheimer's disease MRI datasets:  

- **ADNI**: Alzheimer's Disease Neuroimaging Initiative  
- **OASIS**: Open Access Series of Imaging Studies  
- **AIBL**: Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing
