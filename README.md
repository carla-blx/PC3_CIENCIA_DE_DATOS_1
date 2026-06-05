# PlantVillage - Plant Disease Classification

Clasificación de enfermedades en hojas de plantas usando CNN desde cero y Transfer Learning con MobileNetV2.

## Dataset

[PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) - 21,563 imágenes, 15 clases (3 sanas, 12 enfermas).

## Modelos

| Modelo | Accuracy | Tamaño | Latencia |
|--------|----------|--------|----------|
| CNN desde cero + tuning | **97.0%** | 13.4 MB | 1.5 ms |
| MobileNetV2 + Fine Tuning | 88.9% | 23.7 MB | 72 ms |

## Estructura del proyecto

```
├── notebooks/
│   ├── 01_eda_dataset.ipynb          # Análisis exploratorio
│   ├── 02_preprocesamiento_augmentacion.ipynb
│   ├── 03_cnn_desde_cero.ipynb       # CNN propia + entrenamiento
│   ├── 04_transfer_learning.ipynb    # MobileNetV2 + Fine Tuning
│   ├── 05_ablation_study.ipynb       # Comparativa de modelos
│   └── 06_interpretabilidad.ipynb    # Grad-CAM + métricas
│   ├── modelo_cnn_scratch_final.pth
│   └── modelo_mobilenetv2_final.keras
├── requirements.txt
└── README.md
```

## Requisitos

```bash
pip install -r requirements.txt
```

Principales dependencias: Python 3.8+, PyTorch, TensorFlow, scikit-learn, matplotlib, seaborn.

## Reproducibilidad

- Semilla fija: **SEED = 42**
- División estratificada: 70% train / 15% val / 15% test
- Tamaño de imagen: 224×224

## Resultados clave

- **Mejor modelo**: CNN desde cero con augmentation y tuning → **97.0% accuracy en test**
- **ROC-AUC macro**: 0.9994
- **Grad-CAM** implementado para visualizar regiones relevantes
