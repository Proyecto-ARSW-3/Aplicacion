# AREP — Sistema de Predicción Adaptativa de Deserción Estudiantil

> **Autores:** Juan Pablo Nieto Cortés · Juan Sebastián Buitrago Piñeros · Ángel Cuervo  
> **Institución:** Escuela Colombiana de Ingeniería Julio Garavito  
> **Fecha:** Febrero 2026

Sistema de predicción de deserción estudiantil en instituciones de educación superior,
fundamentado en modelos estadísticos y de aprendizaje automático según el artículo de investigación incluido.

---

## Arquitectura

```
┌─────────────────────┐        REST / JSON         ┌────────────────────────────┐
│   Frontend React    │ ◄─────────────────────────► │  Analytics Service         │
│   (Vite + TS +      │                             │  (Python FastAPI)           │
│    Tailwind +        │                             │                            │
│    Recharts)         │                             │  ┌──────────────────────┐  │
└─────────────────────┘                             │  │   ML Engine          │  │
                                                    │  │  • Logistic Reg.     │  │
                                                    │  │  • Decision Tree     │  │
                                                    │  │  • Random Forest     │  │
                                                    │  │  • SVM               │  │
                                                    │  │  • MLP / Deep (ANN)  │  │
                                                    │  │  • Kaplan-Meier      │  │
                                                    │  │  • Cox PH Model      │  │
                                                    │  └──────────────────────┘  │
                                                    └────────────────────────────┘
```

**Tecnologías elegidas:**

| Capa | Tecnología | Justificación |
|------|-----------|---------------|
| Analytics & ML | Python 3.11 + FastAPI | Integración nativa con scikit-learn y lifelines; el artículo lo menciona explícitamente |
| ML Models | scikit-learn + lifelines | Implementación estándar de todos los modelos del paper |
| Balance de clases | imbalanced-learn (SMOTE) | Recomendado en el paper para datos desbalanceados |
| Frontend | React 18 + TypeScript | Tipo seguro, componentes reutilizables |
| Gráficos | Recharts | Ligero, declarativo, compatible con React |
| Estilos | Tailwind CSS | Utilidades atomizadas, diseño responsivo rápido |
| Contenedores | Docker + Docker Compose | Reproducibilidad y despliegue sencillo |

---

## Estructura del Proyecto

```
AREP Codigo/
├── analytics-service/             # Backend Python FastAPI
│   ├── app/
│   │   ├── main.py                # Entrada de la aplicación + CORS + lifespan
│   │   ├── core/config.py         # Configuración (settings, rutas)
│   │   ├── ml/
│   │   │   ├── preprocessor.py    # Limpieza, codificación, escalado, SMOTE
│   │   │   ├── classifiers.py     # LR, DT, RF, SVM, MLP + métricas
│   │   │   ├── survival.py        # Kaplan-Meier + Cox PH
│   │   │   └── model_manager.py   # Orquestador central (singleton)
│   │   ├── schemas/schemas.py     # Modelos Pydantic de request/response
│   │   ├── services/
│   │   │   ├── analytics_service.py   # Lógica de negocio para analíticas
│   │   │   └── prediction_service.py  # Lógica para predicciones y estudiantes
│   │   └── api/routes/
│   │       ├── analytics.py       # /api/analytics/*
│   │       ├── models.py          # /api/models/*
│   │       ├── predictions.py     # /api/predictions/*
│   │       └── data.py            # /api/data/* (upload, train, status)
│   ├── data/
│   │   ├── generate_sample.py     # Genera dataset sintético si no hay real
│   │   └── dataset.csv            # ← Coloca aquí el dataset de Kaggle
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                      # Frontend React + TypeScript
│   ├── src/
│   │   ├── types/index.ts         # Todos los tipos TypeScript
│   │   ├── services/api.ts        # Cliente REST centralizado
│   │   ├── components/
│   │   │   ├── layout/            # Sidebar, Navbar
│   │   │   ├── charts/            # SurvivalCurve, ModelComparison, VariableImportance, RiskDistribution
│   │   │   └── ui/                # MetricCard, Badge, ThresholdSlider
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx      # KPIs + curva supervivencia + umbral
│   │   │   ├── Students.tsx       # Tabla paginada con filtros
│   │   │   ├── Models.tsx         # Comparativa + importancia de variables
│   │   │   ├── SurvivalAnalysis.tsx  # Kaplan-Meier + tabla de riesgo + Cox
│   │   │   └── Analytics.tsx      # Distribución de riesgo + curva ROC + Youden
│   │   └── App.tsx                # Router principal
│   ├── Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
└── README.md
```

---

## Cómo Ejecutar

### Opción 1: Docker Compose (recomendado)

**Pre-requisito:** Tener Docker Desktop instalado.

```bash
# 1. Clonar / abrir el proyecto
cd "AREP Codigo"

# 2. (Opcional) Colocar el dataset real de Kaggle:
#    Descargar de: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
#    Renombrar a "dataset.csv" y colocar en: analytics-service/data/dataset.csv
#    Si NO lo colocas, se usará un dataset sintético generado automáticamente.

# 3. Construir y levantar
docker-compose up --build

# 4. Abrir el navegador en: http://localhost
```

### Opción 2: Ejecución Local (desarrollo)

**Pre-requisitos:** Python 3.11+, Node.js 20+

**Backend:**
```bash
cd analytics-service

# Instalar dependencias
pip install -r requirements.txt

# Generar dataset de muestra (si no tienes el real)
python data/generate_sample.py

# Levantar el servidor
uvicorn app.main:app --reload --port 8000

# Documentación interactiva disponible en: http://localhost:8000/api/docs
```

**Frontend:**
```bash
cd frontend

# Instalar dependencias
npm install

# Iniciar en modo desarrollo (proxy a localhost:8000)
npm run dev

# Abrir: http://localhost:5173
```

---

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Estado del servicio y si los modelos están entrenados |
| POST | `/api/data/upload` | Subir dataset CSV |
| POST | `/api/data/train` | Iniciar entrenamiento (en background) |
| GET | `/api/data/status` | Estado del entrenamiento |
| GET | `/api/data/info` | Info del dataset cargado |
| GET | `/api/analytics/overview` | KPIs institucionales |
| GET | `/api/analytics/survival` | Datos Kaplan-Meier + Cox |
| GET | `/api/analytics/risk-distribution` | Distribución de probabilidades |
| GET | `/api/analytics/roc` | Curva ROC (con `?model_name=`) |
| GET | `/api/models/comparison` | Métricas de todos los modelos |
| GET | `/api/models/feature-importance` | Importancia de variables |
| GET | `/api/models/list` | Lista de modelos con mejor modelo marcado |
| GET | `/api/predictions/students` | Estudiantes con riesgo (paginado, filtrable) |
| POST | `/api/predictions/threshold` | Actualizar umbral τ |
| POST | `/api/predictions/predict` | Predecir un estudiante nuevo |

---

## Dataset

El sistema fue validado con el dataset **"Predict Students' Dropout and Academic Success"**:
- **Fuente:** Instituto Politécnico de Portalegre (2022), UCI Machine Learning Repository
- **Kaggle:** https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
- **Registros:** 4,424 estudiantes universitarios
- **Variables:** 35 atributos (académicos, socioeconómicos, demográficos, macroeconómicos)
- **Variable objetivo:** `Target` → Dropout / Graduate / Enrolled

### Cómo usar el dataset real

1. Descarga el archivo desde Kaggle (requiere cuenta gratuita)
2. Renómbralo a `dataset.csv`
3. Colócalo en `analytics-service/data/dataset.csv`
4. El sistema lo detecta automáticamente al iniciar y entrena los modelos

Si no colocas el dataset real, `generate_sample.py` crea datos sintéticos con la misma estructura.

---

## Modelos Implementados

| Modelo | Librería | Hiperparámetros principales |
|--------|---------|---------------------------|
| Regresión Logística | sklearn | C=1.0, max_iter=1000 |
| Árbol de Decisión | sklearn | max_depth=8, min_samples_split=20 |
| Random Forest | sklearn | n_estimators=100, max_depth=10, n_jobs=-1 |
| SVM | sklearn | kernel=rbf, C=1.0, probability=True |
| Red Neuronal (MLP) | sklearn | hidden=(128,64,32), activation=relu, early_stopping |
| Kaplan-Meier | lifelines | No paramétrico |
| Cox PH | lifelines | penalizer=0.1 |

**Pipeline de preprocesamiento:**
1. Imputación de valores faltantes (mediana / moda)
2. Escalado Z-score (StandardScaler)
3. SMOTE para balance de clases (solo en entrenamiento)
4. Split 80% train / 20% test, estratificado

**Umbral adaptativo:** Calculado por índice de Youden (τ* = argmax[Sensibilidad + Especificidad − 1])

---

## Páginas del Frontend

| Página | Ruta | Contenido |
|--------|------|-----------|
| Dashboard | `/` | KPIs, curva de supervivencia, distribución de riesgo, control de umbral |
| Estudiantes | `/students` | Tabla paginada con filtros por nivel de riesgo y umbral personalizable |
| Modelos ML | `/models` | Comparativa de métricas, selección de métrica, importancia de variables |
| Supervivencia | `/survival` | Curva K-M, tabla de riesgo por semestre, hazard ratios Cox |
| Analíticas | `/analytics` | Distribución de probabilidades, curva ROC, tabla índice Youden |

---

## Dependencias Clave

**Backend:**
- `fastapi` — framework web async
- `scikit-learn` — modelos ML
- `lifelines` — análisis de supervivencia
- `imbalanced-learn` — SMOTE
- `pandas` + `numpy` — procesamiento de datos

**Frontend:**
- `react` + `react-router-dom` — SPA
- `recharts` — gráficos
- `tailwindcss` — estilos utilitarios
- `lucide-react` — iconos
- `clsx` — manejo condicional de clases CSS
