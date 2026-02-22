# Proyecto 7 — Predicción de carreras aspiracionales en Gen Z (Multilabel)

> **Objetivo**: predecir el **Top‑3** de carreras aspiracionales (de un catálogo de 10) a partir de respuestas de una encuesta sobre valores y preferencias laborales de la Generación Z.

---

## 1) Contexto
La literatura y reportes de mercado suelen describir a la Gen Z como una cohorte que prioriza **equilibrio vida‑trabajo**, **propósito**, **bienestar** y mayor **exigencia de coherencia** entre discurso y prácticas organizacionales. En este proyecto, usamosse usaron respuestas de una encuesta para **inferir** qué “familias de carrera” se alinean con esos atributos.

**Pregunta de negocio**  
> Dado el perfil de respuestas de una persona, ¿cuáles son las **3** carreras aspiracionales más probables?

---

## 2) Datos
- **Dataset**: `Your Career Aspirations of GenZ.csv`.  
- **Observaciones**: **235** encuestas (filas).
- **Variables**: **15** columnas originales.
  - **Features usadas por el modelo**: **13** (mayoritariamente categóricas + 1 variable numérica).
  - **Exclusión**: `Your Current Zip Code / Pin Code` (alta cardinalidad: **190** valores únicos sobre 235, aporta ruido y poco poder predictivo en este tamaño muestral).

Fuente del dataset: Kaggle (KultureHire). Ver referencias.

---

## 3) Target (multilabel)
- Columna objetivo: `Which of the below careers looks close to your Aspirational job ?`
- Cada fila contiene **3 etiquetas** (por eso es **multilabel**).
- Catálogo de **10** carreras:
  1. Become a content Creator in some platform  
  2. Build and develop a Team  
  3. Business Operations in any organization  
  4. Design and Creative strategy in any company  
  5. Design and Develop amazing software  
  6. Look deeply into Data and generate insights  
  7. Manage and drive End-to-End Projects or Products  
  8. Teaching in any of the institutes/online or Offline  
  9. Work as a freelancer and do my thing my way  
  10. Work in a BPO setup for some well known client  

---

## 4) EDA (hallazgos clave)
1) **Sesgo geográfico**: el dataset está fuertemente concentrado en **India** (ver gráficos en el notebook / PPT).  
2) **Alta cardinalidad** en ZIP/PIN: explica por qué se elimina esa variable para evitar sobreajuste.

---

## 5) Preprocesamiento
Pipeline (aplicado a features):
- **Normalización de texto**: limpieza básica (espacios, consistencia, etc.).
- **Agrupación de categorías raras** → `"Other"` usando umbral **2%** (`min_freq=0.02`).
- **One‑Hot Encoding** (categóricas) con `handle_unknown="ignore"` para robustez en despliegue.

---

## 6) Modelo (baseline)
- **Estrategia multilabel**: **One‑Vs‑Rest** (entrena 10 clasificadores binarios, uno por carrera).  
- **Clasificador base**: **Logistic Regression** con `class_weight="balanced"` para mitigar desbalance.  
- **Split**: Train/Test **80/20** (188/47).

---

## 7) Métricas (explicación simple)
- **Subset accuracy (exact match)**: acierto sólo si predice **exactamente** las 3 carreras correctas → métrica exigente en multilabel.
- **F1 micro**: mide desempeño global ponderando por frecuencia.
- **F1 macro**: promedio simple por carrera (penaliza más a clases raras).
- **Jaccard micro**: similitud entre conjuntos de etiquetas (predicho vs real).
- **Precision@3 / Recall@3**: calidad de las **Top‑3** sugerencias.

---

## 8) Resultados
| Modelo | Subset Acc | F1 micro | F1 macro | Jaccard micro | Prec@3 | Rec@3 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (OvR LR) | 0.0000 | 0.3670 | 0.3227 | 0.2247 | — | — |
| **Tuned (OvR LR)** | **0.0213** | **0.4073** | **0.3555** | **0.2557** | **0.3688** | **0.3688** |
| Ensamble (tuned + SGD) | 0.0000 | 0.3300 | 0.2563 | 0.1976 | 0.3191 | 0.3191 |

**Decisión**: el **modelo tuneado** es el candidato para demo (mejor equilibrio global en métricas).

---

## 9) Demo API (esta sección es solo descriptiva indica lo que se debería hacer, para que cualquier usuario pueda mandar datos y que esta misma devuelva la predicción del modelo.)
> Nota:El notebook no conserva el **URL público** (ngrok). Se debería generar al levantar la API y **reemplazarlo** en la presentación/README.

### Estructura del request (JSON)
```json
{
  "data": {
    "Your Current Country.": "India",
    "Your Gender": "Male",
    "Which of the below factors influence the most about your career aspirations ?": "People who have changed the world for better",
    "Would you definitely pursue a Higher Education / Post Graduation outside of India ? If only you have to self sponsor it.": "Yes, I will earn and do that",
    "How likely is that you will work for one employer for 3 years or more ?": "This will be hard to do, but if it is the right company I will",
    "Would you work for a company whose mission is not clearly defined and publicly posted.": "No",
    "How likely would you work for a company whose mission is misaligned with their public actions or even their product ?": "Will NOT work for them",
    "How likely would you work for a company whose mission is not bringing social impact ?": 4,
    "What is the most preferred working environment for you.": "Hybrid",
    "Which of the below Employers would you work with.": "Google",
    "Which type of learning environment that you are most likely to work in ?": "Practical training with real world projects and case studies",
    "What type of Manager would you work without looking into your watch ?": "A manager who empowers you and enables you to take educated risks",
    "Which of the following setup you would like to work ?": "Small Team working on a module"
  }
}
```

### Ejemplo (cURL) — reemplaza la URL
```bash
curl -X POST https://TU_URL_PUBLICA.ngrok-free.app/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

---

## 10) Entregables
- `GenZ_Project7_Multilabel_Presentacion.pptx`.
- Notebook (`.ipynb`) con EDA/modelado.
- README.

---

## 11) Limitaciones
- Dataset **pequeño** (235) y **sesgado por país** → generalización limitada.
- Etiquetas desbalanceadas (p.ej. “BPO” muy rara) → macro‑F1 penaliza.
- **Subset accuracy** tiende a ser baja en multilabel Top‑3 (métrica muy estricta).
- Se excluye ZIP por alta cardinalidad: se pierde señal geográfica fina.

---

## 12) Próximos pasos
1) Aumentar muestra y balancear países/segmentos.  
2) Tratar ZIP con **hashing/embeddings** o agrupar por región.  
3) Probar modelos multilabel alternativos: **Classifier Chains**, calibración de probabilidades, umbrales por clase.  
4) Validación con **cross‑validation** y análisis de estabilidad.

---

## Referencias (APA)
- Deloitte. (2025, June 2). *2025 Gen Z and millennial survey*. Deloitte Insights. https://www.deloitte.com/us/en/insights/topics/talent/2025-gen-z-millennial-survey.html.
- Deloitte. (2025, May 14). *2025 Gen Z and Millennial Survey* (global hub). https://www.deloitte.com/global/en/issues/work/genz-millennial-survey.html.
- KultureHire. (s. f.). *Understanding Career Aspirations of GenZ* [Dataset]. Kaggle. https://www.kaggle.com/datasets/kulturehire/understanding-career-aspirations-of-genz/data.
- Pew Charitable Trusts. (2023, December 8). *It’s time to create mentally healthy workplaces*. https://www.pew.org/en/trend/archive/fall-2023/its-time-to-create-mentally-healthy-workplaces.
- scikit‑learn developers. (s. f.). *accuracy_score* — scikit‑learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html.
- scikit‑learn developers. (s. f.). *f1_score* — scikit‑learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html.
- scikit‑learn developers. (s. f.). *jaccard_score* — scikit‑learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html.
- scikit‑learn developers. (s. f.). *OneVsRestClassifier* — scikit‑learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html.
- scikit‑learn developers. (s. f.). *LogisticRegression* — scikit‑learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
