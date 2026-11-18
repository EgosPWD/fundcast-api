# Startup Funding API: contexto del proyecto

Servicio web con API REST que estima el monto (USD) de una ronda de financiación de una startup.
Trabajo final de la materia de IA. Idioma de documentos: español. Código y nombres técnicos: inglés.
El cliente web vive en **otro repositorio**; este repo solo expone la API.

## 1. Qué ya está hecho (NO rehacer)
Un notebook de Colab (`notebooks/Startups_Monto_CRISP_ML.ipynb`) ya cubre exploración, limpieza, variables, comparación de 13 modelos, ajuste de hiperparámetros, evaluación y monitoreo. Tu trabajo es convertir esa lógica en un servicio.

- Dataset: Kaggle `fliarbi/global-startup-funding-rounds-2019-2026`, archivo `global_startup_funding_sample.csv`, 20.227 filas y 19 columnas. Licencia CC BY-NC (solo uso académico).
- Tarea: regresión supervisada. Objetivo: `log10(amount_usd)`.
- Modelo final: `HistGradientBoostingRegressor` dentro de un `Pipeline` de scikit-learn.
  Parámetros ganadores (GridSearch): `learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=10`.
- Métricas en test (3.190 filas): R² 0,620 · MAE_log 0,352 · RMSE_log 0,494 · MdAPE 52,5 % · cobertura ±50 % = 47,9 % · línea base (mediana) R² 0.
- Artefacto del modelo: un único archivo, `artifacts/modelo_rondas.joblib` (generado por el notebook, versionado en git como modelo base). Es un `dict` con las claves `"pipeline"`, `"columns"`, `"restricciones"`, `"metricas_test"` y `"parametros"`.
- El R² real es ≈ 0,62. No intentes "subir el puntaje" con trucos.

## 2. Reglas de datos que NO se pueden violar
1. **Prohibido usar `news_excerpt`** como variable: contiene el monto escrito (fuga de datos, 62 % de las rondas).
2. **Filas de entrenamiento:** `amount_status == "verified"`, `amount_usd` no nulo y `amount_usd >= 1000`. (`failed_check` y `not_disclosed` no tienen monto.)
3. **Orden:** partición 80/20 aleatoria (`random_state=42`) ANTES de ajustar transformadores. Todo transformador va dentro del `Pipeline`.
4. El objetivo se entrena en `log10`; **la API devuelve USD** (`10 ** pred`) y también el log.
5. Las variables de historial (`prev_log_amount`) usan solo rondas ANTERIORES de la misma empresa.

## 3. Entrada del modelo (contrato exacto)
El `Pipeline` espera un `DataFrame` con **estas 13 columnas** (nombres exactos; los nulos se aceptan y el pipeline los imputa):

| Columna | Tipo | Notas |
|---|---|---|
| `year` | int | año del anuncio, 2019–2026 |
| `company_age` | int | `year - founded_year` (la API la calcula) |
| `investor_count` | int | número de inversores |
| `news_mentions` | int | menciones en prensa |
| `has_lead_investor` | 0/1 | 1 si hay `lead_investor` (la API la calcula) |
| `prev_log_amount` | float, nullable | `log10` del monto de la ronda anterior de esa empresa |
| `stage_rank` | float, nullable | orden de la etapa (ver mapa); `Undisclosed`, `Debt`, `Bridge` = null |
| `stage` | str | etapa de la ronda |
| `sector` | str | |
| `country` | str | |
| `city` | str | normalizada |
| `lead_investor` | str, nullable | |
| `news_source` | str, nullable | |

Mapa `stage_rank`: Grant/Angel/Crowdfunding=0, Pre-seed=1, Seed=2, Pre-Series A=3, Series A=4, Pre-Series B=5, Series B=6, Pre-Series C=7, Series C=8, Series D=10, E=11, F=12, G=13, H=14.

**Cuerpo de `POST /predict`** (lo que envía el cliente; la API deriva el resto):
```json
{
  "year": 2025, "founded_year": 2021, "stage": "Seed", "sector": "Fintech",
  "country": "United States", "city": "New York",
  "investor_count": 3, "lead_investor": "Y Combinator",
  "news_mentions": 2, "news_source": "TechCrunch",
  "prev_amount_usd": 1500000
}
```
Obligatorios: `year, founded_year, stage, sector, country`. El resto es opcional.
Derivaciones: `company_age = year - founded_year`; `has_lead_investor = lead_investor is not None`;
`prev_log_amount = log10(prev_amount_usd)` o null; `stage_rank = mapa[stage]` o null.

**Respuesta:**
```json
{ "prediction_id": 12, "estimated_amount_usd": 4150000, "estimated_log10": 6.62,
  "range_usd": [2100000, 8200000], "model_version": 3 }
```
`range_usd` = estimado × / ÷ 1,5 (≈ error mediano del 50 %). Avisar si `stage == "Undisclosed"`: el modelo falla más ahí (MdAPE ≈ 67 %).

## 4. Limpieza que `train.py` debe reproducir (en este orden)
1. `announced_date` a fecha; comprobar `year == announced_date.year`.
2. Texto: quitar espacios en `country, city, sector, stage, company_name, lead_investor`; unificar variantes de `city` (clave en minúsculas sin símbolos → valor más frecuente).
3. Banderas (no se eliminan filas): `amount_suspect_low` (<1000), `amount_outlier_high` (IQR sobre log10), `dup_same_amount`.
4. Variables: `company_age`, `log_amount`, `has_lead_investor`, `n_prev_rounds`, `prev_log_amount` (último monto conocido de la empresa, ordenado por fecha, con `shift(1).ffill()`), `stage_rank`.
5. Transformador (`ColumnTransformer`):
   - numéricas: `SimpleImputer(median, add_indicator=True)` + `StandardScaler`;
   - categóricas: `SimpleImputer(constant "Unknown")` + `TargetEncoder(random_state=42)`.
6. Modelo: `HistGradientBoostingRegressor(random_state=42, **mejores_parametros)`.

Para `/retrain`: reentrenar con el CSV original más el archivo subido (misma limpieza), SIN repetir la búsqueda de hiperparámetros (usar los parámetros ganadores). Evaluar en test reservado.
Activar la nueva versión solo si `MAE_log_nuevo <= MAE_log_activo + 0.01`; si no, guardarla como inactiva y avisar.

## 5. Arquitectura hexagonal
```
fundcast-api/
├── AGENTS.md
├── CLAUDE.md                    # solo enlaza a AGENTS.md
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .dockerignore
├── alembic.ini
├── alembic/versions/
├── app/
│   ├── main.py                  # composición: conecta puertos y adaptadores, crea FastAPI
│   ├── config.py                # settings desde variables de entorno (pydantic-settings)
│   ├── domain/                  # entities.py, errors.py
│   ├── application/
│   │   ├── ports/               # predictor, trainer, prediction_repo, model_repo, user_repo,
│   │   │                        # training_file_repo, file_storage, password_hasher, token_service
│   │   └── use_cases/           # register_user, login, predict_round, list_predictions,
│   │                            # get_metrics, upload_training_file, retrain_model
│   └── adapters/
│       ├── inbound/api/         # routers/, schemas.py, security.py, deps.py
│       └── outbound/            # ml/ (cleaning, features, sklearn_trainer, sklearn_predictor),
│                                # postgres/ (models, session, *_repo), storage/
├── tests/                       # unit/ (repos falsos en memoria), integration/ (TestClient)
├── artifacts/                   # modelo_rondas.joblib en git; los que genera /retrain, git-ignored
├── data/uploads/                # CSV subidos (git-ignored)
├── db/                          # schema.sql y diagrama ER
├── docs/                        # openapi.json
└── notebooks/                   # notebook de Colab (solo referencia)
```
**Regla de dependencias:** `domain` no importa nada; `application` solo importa `domain` y sus puertos; los routers solo llaman a casos de uso; nada fuera de `adapters/outbound` importa SQLAlchemy, scikit-learn o joblib.

## 6. Endpoints
| Método | Ruta | Rol | Descripción |
|---|---|---|---|
| POST | `/auth/register` | público | crea usuario (rol `user`) |
| POST | `/auth/login` | público | devuelve JWT |
| POST | `/predict` | user, admin | predice y guarda en el historial |
| GET | `/predictions` | user (solo las suyas), admin (todas) | historial: datos enviados y resultado; paginado |
| GET | `/metrics` | user, admin | métricas de la versión activa e historial de versiones |
| POST | `/training-files` | admin | sube CSV de entrenamiento |
| POST | `/retrain` | admin | reentrena (último archivo o `file_id`) |
| GET | `/health` | público | estado |

Errores: 401 sin token, 403 sin permiso, 422 entrada inválida, 409 si ya hay un reentrenamiento en curso.
Validar el CSV subido: columnas obligatorias (sección 13), tamaño máximo 20 MB, extensión `.csv`; nunca ejecutar nada de su contenido.
Swagger automático en `/docs`; exportar `docs/openapi.json`.

## 7. Base de datos (PostgreSQL, migraciones con Alembic)
- `users(id, email UNIQUE, password_hash, role, created_at)`
- `model_versions(id, version INT UNIQUE, artifact_path, is_active BOOL, mae_log, r2, rmse_log, mdape, coverage, n_train, n_test, params JSONB, training_file_id FK NULL, created_at)`
- `training_files(id, filename, stored_path, rows, uploaded_by FK users, status, created_at)`
- `predictions(id, user_id FK, model_version_id FK, input JSONB, estimated_amount_usd, estimated_log10, created_at)`

Generar `db/schema.sql` y el diagrama ER (dbdiagram.io o mermaid).

## 8. Seguridad y configuración
Variables (`.env`, nunca en git): `DB_PASSWORD`, `DATABASE_URL`, `JWT_SECRET`, `JWT_ALGORITHM=HS256`, `JWT_EXPIRE_MINUTES`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ARTIFACTS_DIR`, `UPLOADS_DIR`, `CORS_ORIGINS`.
Contraseñas con bcrypt. Crear el admin al arrancar si no existe.
**`joblib.load` solo sobre artefactos propios del proyecto** (`modelo_rondas.joblib` del notebook y los que genera `/retrain` en `ARTIFACTS_DIR`), nunca sobre archivos subidos por usuarios (deserialización insegura).
scikit-learn está fijado en **`1.6.1`** (`requirements.txt`), la misma versión con la que se entrenó `modelo_rondas.joblib`. No cambiarla sin regenerar el modelo (`TargetEncoder` requiere ≥ 1.3).

## 9. Pruebas mínimas (pytest)
Casos de uso con repositorios falsos; login y permisos (403 a `user` en `/retrain`); `/predict` con campos mínimos y con nulos; categoría desconocida no rompe; historial filtrado por usuario; subida de CSV inválido devuelve 422.

## 10. Cliente web (fuera de este repo)
La interfaz vive en un repositorio separado. Este repo solo expone la API. La API debe garantizar:
- CORS configurable con `CORS_ORIGINS` (lista separada por comas).
- Contrato estable y documentado en `docs/openapi.json`; no romper rutas ni campos sin avisar.
- Errores en JSON con la forma `{"detail": "..."}` y códigos HTTP correctos.
- JWT en el header `Authorization: Bearer <token>`.

No hay carpeta `frontend/` ni se sirven archivos estáticos desde aquí.

## 11. Entregables de este repo
Código de la API · `db/schema.sql` y diagrama ER · `docs/openapi.json` (Swagger) · README con instrucciones · pruebas · repositorio GitHub público · evidencia de revisión de PR con IA (plus).
El cronograma, el documento de CRISP-ML(Q), el video y la interfaz se entregan aparte y NO se generan aquí.

## 12. Orden de trabajo
1. `domain` + `ports`.
2. `cleaning.py`, `features.py`, `sklearn_trainer.py`, `sklearn_predictor.py` (salen del notebook; reproducir R² ≈ 0,62).
3. Casos de uso con pruebas.
4. PostgreSQL + Alembic.
5. FastAPI, JWT y roles.
6. Docker Compose.
7. `openapi.json` y README.

Ramas con Pull Requests pequeños (uno por paso).

### Flujo de Git (Git Flow)
- `main`: solo versiones estables y entregables. Nunca se hace commit directo.
- `develop`: rama de integración. Todo trabajo nuevo sale de aquí y vuelve aquí.
- `feature/<nombre>`: una por paso del orden de trabajo con un nombre que describa lo que entrega (ej. `feature/model-inference-adapter`, `feature/postgres-persistence-and-migrations`). Sale de `develop` y vuelve por Pull Request a `develop`.
- `release/<versión>`: sale de `develop` para preparar una entrega; se fusiona en `main` (con tag `vX.Y.Z`) y de vuelta en `develop`.
- `hotfix/<nombre>`: sale de `main` para corregir algo urgente; se fusiona en `main` y en `develop`.

### Commits (Conventional Commits)
Formato: `<tipo>(<alcance opcional>): <descripción en español, imperativo, minúscula>`. El `<tipo>` y el alcance quedan en inglés (son parte del estándar y nombres técnicos).
- Tipos: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `build`, `ci`, `perf`, `style`.
- Ejemplos: `feat(domain): agrega la entidad de ronda de financiación`, `test(use-cases): cubre la predicción con campos nulos`, `build(docker): fija la imagen base python 3.12`.
- Cambios incompatibles en el contrato de la API: `feat!:` o pie `BREAKING CHANGE:` (ver sección 10).
- Commits pequeños y coherentes: cada uno con su código y sus pruebas.
- Sin atribuciones de IA (`Co-Authored-By` u otras) en los mensajes.

### Pull Requests
- Título y descripción **en español** (el título sigue el formato de Conventional Commits, ej. `feat(ml): adaptador de predicción con el modelo base`).
- Cada PR pide revisión con IA comentando `@coderabbitai review`; los hallazgos válidos se corrigen, se responde en el hilo y se resuelve antes de fusionar.

## 13. Columnas del CSV (verificadas)
Columnas reales de `global_startup_funding_sample.csv` (19):
`round_id`, `announced_date`, `year`, `company_name`, `company_domain`, `country`, `city`, `sector`, `founded_year`, `stage`, `amount_usd`, `amount_status`, `lead_investor`, `investor_count`, `news_mentions`, `news_source`, `news_url`, `news_excerpt`, `fliar_profile_url`.

**Obligatorias en un CSV para `/retrain`:** `announced_date`, `year`, `company_domain`, `founded_year`, `stage`, `sector`, `country`, `city`, `lead_investor`, `investor_count`, `news_mentions`, `news_source`, `amount_usd`, `amount_status`.
- `company_domain` es obligatoria: con ella se calcula el monto de la ronda anterior de cada empresa (`prev_log_amount`).
- Opcionales (no entran al modelo): `round_id`, `company_name`, `news_url`, `fliar_profile_url`.
- **NUNCA usar `news_excerpt` como variable** (ver sección 2).
