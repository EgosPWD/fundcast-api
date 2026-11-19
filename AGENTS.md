# Startup Funding API: contexto del proyecto

Servicio web con API REST que estima el monto (USD) de una ronda de financiación de una startup.
Trabajo final de la materia de IA. Idioma de documentos: español. Código y nombres técnicos: inglés.
El cliente web vive en **otro repositorio**; este repo solo expone la API.

## 1. Qué ya está hecho (NO rehacer)
Un notebook de Colab (`notebooks/Startups_Monto_CRISP_ML.ipynb`) ya cubre exploración, limpieza, variables, comparación de 13 modelos, ajuste de hiperparámetros, evaluación y monitoreo. Tu trabajo es convertir esa lógica en un servicio.

- Dataset: Kaggle `fliarbi/global-startup-funding-rounds-2019-2026`, archivo `global_startup_funding_sample.csv`, 20.227 filas y 19 columnas. Licencia CC BY-NC (solo uso académico).
- Tarea: regresión supervisada. Objetivo: `log10(amount_usd)`.
- Modelo final: `HistGradientBoostingRegressor` dentro de un `Pipeline` de scikit-learn.
  Parámetros ganadores (GridSearch): `learning_rate=0.03, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40`.
- Partición: 80/20 **estratificada por deciles** de `log10(amount_usd)` (`random_state=42`): 12.758 filas de entrenamiento y 3.190 de test.
- Métricas en test (3.190 filas): R² 0,615 · MAE_log 0,350 · RMSE_log 0,495 · MdAPE 52,9 % · cobertura ±50 % = 47,3 % · sobreestima 51,0 % · línea base (mediana) MAE_log 0,629.
- Criterios de aceptación (clave `criterios` del artefacto): R² ≥ 0,55 y MAE_log ≤ 0,40. El modelo los cumple.
- Artefacto del modelo: `artifacts/modelo_rondas.joblib` (1,3 MB, generado por el notebook, versionado en git como modelo base). Es un `dict` con las claves `"pipeline"`, `"columns"`, `"criterios"`, `"metricas_test"`, `"parametros"`, `"lead_freq_map"` (inversor → número de rondas que lideró, 8.463 inversores) y `"stage_rank_map"` (etapa → orden).
- El entrenamiento está **verificado**: con la limpieza de la sección 4 se reproducen exactamente las 6 métricas sobre las mismas 3.190 filas.
- El R² real es ≈ 0,615. No intentes "subir el puntaje" con trucos.

## 2. Reglas de datos que NO se pueden violar
1. **Prohibido usar `news_excerpt`** como variable: contiene el monto escrito (fuga de datos, 62 % de las rondas). Ni siquiera se guarda en la base de datos.
2. **Filas de entrenamiento:** `amount_status == "verified"`, `amount_usd` no nulo y `amount_usd >= 1000`. (`failed_check` y `not_disclosed` no tienen monto.)
3. **Orden:** partición 80/20 estratificada (`random_state=42`) ANTES de ajustar transformadores. Todo transformador va dentro del `Pipeline`.
4. El objetivo se entrena en `log10`; **la API devuelve USD** (`10 ** pred`) y también el log.
5. Las variables de historial (`n_prev_rounds`, `prev_log_amount`) usan solo rondas ANTERIORES de la misma empresa.

## 3. Entrada del modelo (contrato exacto)
El `Pipeline` espera un `DataFrame` con **estas 14 columnas**, en este orden (los nulos se aceptan y el pipeline los imputa):

| Columna | Tipo | Notas |
|---|---|---|
| `year` | int | año del anuncio, 2019–2026 |
| `company_age` | int | `year - founded_year` (la API la calcula) |
| `investor_count` | int, nullable | número de inversores |
| `news_mentions` | int, nullable | menciones en prensa |
| `n_prev_rounds` | int | número de rondas ANTERIORES de la empresa |
| `prev_log_amount` | float, nullable | `log10` del último monto conocido de una ronda anterior de esa empresa |
| `lead_freq` | int, nullable | número de rondas que lideró `lead_investor`; null si no hay líder |
| `stage_rank` | float, nullable | orden de la etapa (ver mapa); `Undisclosed`, `Debt`, `Bridge` = null |
| `stage` | str | etapa de la ronda |
| `sector` | str | |
| `country` | str | |
| `city` | str, nullable | normalizada |
| `lead_investor` | str, nullable | |
| `news_source` | str, nullable | |

`has_lead_investor` **ya no** es una variable del modelo.

Mapa `stage_rank` (textos exactos, tras el `strip`; el resto de etapas queda null): Grant/Angel/Crowdfunding=0, Pre-seed=1, Seed=2, Pre-Series A=3, Series A=4, Pre-Series B=5, Series B=6, Pre-Series C=7, Series C=8, Series D=10, Series E=11, Series F=12, Series G=13, Series H=14.

**Cuerpo de `POST /predict`** (lo que envía el cliente; la API deriva el resto):
```json
{
  "year": 2025, "founded_year": 2021, "stage": "Seed", "sector": "Fintech",
  "country": "United States", "city": "New York",
  "investor_count": 3, "lead_investor": "Y Combinator",
  "news_mentions": 2, "news_source": "TechCrunch",
  "company_domain": "acme.io", "prev_rounds": 1, "prev_amount_usd": 1500000
}
```
Obligatorios: `year, founded_year, stage, sector, country`. El resto es opcional.

Derivaciones:
- `company_age = year - founded_year`; `stage_rank = stage_rank_map[stage]` o null.
- Si llega `company_domain` **y existe** en `funding_rounds`: `n_prev_rounds` y `prev_log_amount` se calculan desde la tabla, con las rondas de esa empresa cuyo `year <= year` de la predicción (supuesto: todas las rondas conocidas hasta ese año son "anteriores"). `prev_log_amount` = `log10` del último monto > 0 por `announced_date`.
- Si no: `n_prev_rounds = prev_rounds` o 0; `prev_log_amount = log10(prev_amount_usd)` o null.
- `lead_freq = lead_freq_map.get(lead_investor, 0) + 1` (el `+1` es porque en el entrenamiento la propia ronda cuenta), usando el `lead_freq_map` del modelo **activo**; null si no hay `lead_investor`.

**Respuesta:**
```json
{ "prediction_id": 12, "estimated_amount_usd": 4150000, "estimated_log10": 6.62,
  "range_usd": [2100000, 8200000], "model_version": 3, "warning": null }
```
`range_usd` = estimado × / ÷ 1,5 (≈ error mediano del 50 %). `warning` avisa si `stage == "Undisclosed"`: el modelo falla más ahí (MdAPE ≈ 71 %).

## 4. Limpieza y entrenamiento (reproducen el notebook; verificado)
Se aplican sobre **todas** las filas de `funding_rounds` (CSV original + cargas), leídas **siempre ordenadas por `id`** (el orden del CSV original). Sin ese orden fijo, la partición no es reproducible.

1. Fecha: `announced_date = pd.to_datetime(announced_date)`; se exige `announced_date.year == year` (no se corrige ni se borra; en una carga, si no coincide → 422).
2. Texto: solo `strip` y colapsar espacios, **sin cambiar mayúsculas**, en `country, city, sector, stage, company_name, lead_investor`:
   `s.str.strip().str.replace(r"\s+", " ", regex=True)`.
3. `city`: clave = minúsculas sin símbolos (`str.lower().str.replace(r"[^a-z0-9]", "", regex=True)`); cada clave toma su valor más frecuente con `mode()[0]` (empate: el primero en orden alfabético). Los nulos siguen nulos.
4. Banderas (sobre todas las filas; no borran nada): `amount_suspect_low = amount_usd < 1000`; `amount_outlier_high = amount_usd > 10 ** (q3 + 1.5 * (q3 - q1))` (cuartiles de `log10(amount_usd)` con monto > 0); `dup_same_amount` = duplicados en `company_domain, stage, amount_usd` (`keep=False`) con monto no nulo y `stage != "Undisclosed"`. **Solo `amount_suspect_low` excluye filas del entrenamiento.**
5. Variables, sobre **todas** las filas y ANTES de filtrar `verified`:
   - `log_amount = log10(amount_usd)` si monto > 0.
   - `history = df.sort_values(["company_domain", "announced_date"]).groupby("company_domain")` (clave tal cual, sin desempate por `round_id`).
   - `prev_log_amount = history["log_amount"].transform(lambda s: s.shift(1).ffill())`. Puede venir de una ronda no `verified`; se replica así para reproducir el 0,615 (mejora futura, NO ahora).
   - `n_prev_rounds = history.cumcount()`.
   - `lead_freq = lead_investor.map(lead_investor.value_counts())` (cuenta la propia fila; null si no hay líder).
   - `company_age = year - founded_year`; `stage_rank` con el mapa de la sección 3.
6. Filas usables: `amount_status == "verified"` & `amount_usd` no nulo & `~amount_suspect_low`.
7. Partición: `train_test_split(X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=10, labels=False))`, con `y = log_amount`.
8. Transformador (`ColumnTransformer`, `remainder="drop"`):
   - `"num"`: `SimpleImputer(strategy="median", add_indicator=True)` + `StandardScaler()`;
   - `"cat"`: `SimpleImputer(strategy="constant", fill_value="Unknown")` + `TargetEncoder(random_state=42)` (resto por defecto).
9. Modelo: `Pipeline([("prep", transformador), ("model", HistGradientBoostingRegressor(random_state=42, **parametros_ganadores))])`.
10. Métricas: R², MAE y RMSE sobre `log10`; sobre USD (`real = 10**y`, `est = 10**pred`, `ape = |est - real| / real`): `MdAPE = median(ape)`, `cobertura = mean(ape <= 0.5)`, `pct_sobreestima = mean(est > real)`.

### Flujo de `/retrain` (requisito del docente)
Cada reentrenamiento hace una **partición 80/20 estratificada NUEVA** sobre todas las filas de `funding_rounds` en ese momento, entrena con los parámetros ganadores (SIN repetir la búsqueda de hiperparámetros) y evalúa en ese nuevo test.
- Con solo el CSV original en la tabla, el reentrenamiento reproduce exactamente el modelo base (R² 0,615).
- Activación: la nueva versión se activa solo si `MAE_log_nuevo <= MAE_log_activo + 0.01`, comparando con el MAE **guardado** de la versión activa; si no, se guarda como inactiva y se avisa.
- **Límite a documentar:** cada versión se mide sobre su propio test, así que una diferencia pequeña de MAE puede deberse al sorteo y no solo al modelo.
- El nuevo artefacto guarda las mismas claves que el base, con `lead_freq_map` recalculado sobre toda la tabla.

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
│   │   │                        # training_file_repo, funding_round_repo, file_storage,
│   │   │                        # password_hasher, token_service, retrain_lock, model_inspector
│   │   └── use_cases/           # register_user, login, ensure_admin, predict_round,
│   │                            # list_predictions, get_metrics, upload_training_file,
│   │                            # retrain_model, ensure_baseline_model, seed_funding_rounds
│   └── adapters/
│       ├── inbound/api/         # routers/, schemas.py, security.py, deps.py
│       └── outbound/            # ml/ (cleaning, features, sklearn_trainer, sklearn_predictor),
│                                # postgres/ (models, session, *_repo), storage/
├── tests/                       # unit/ (repos falsos en memoria), integration/ (TestClient)
├── artifacts/                   # modelo_rondas.joblib en git; los que genera /retrain, git-ignored
├── data/                        # CSV original de Kaggle (git-ignored, licencia CC BY-NC)
│   └── uploads/                 # copia de los CSV subidos (git-ignored)
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
| POST | `/training-files` | admin | sube un CSV: se valida y sus filas se insertan en `funding_rounds` |
| POST | `/retrain` | admin | reentrena con **todas** las filas de `funding_rounds` (partición nueva) |
| GET | `/health` | público | estado |

Errores: 401 sin token, 403 sin permiso, 422 entrada inválida, 409 si ya hay un reentrenamiento en curso.
Validar el CSV subido: columnas obligatorias (sección 13), tamaño máximo 20 MB, extensión `.csv`, `year == año de announced_date`; nunca ejecutar nada de su contenido. Las filas con `company_age` negativa o `founded_year` nulo se rechazan (se cuentan en la respuesta). Las filas con un `round_id` que ya existe se ignoran.
Swagger automático en `/docs`; exportar `docs/openapi.json`.

### Arranque
- Si `funding_rounds` está vacía y `SEED_DATASET_PATH` apunta a un CSV, se carga ese CSV (el original de Kaggle) en el orden del archivo.
- Si `model_versions` está vacía, `modelo_rondas.joblib` se registra como versión 1 activa (la API predice desde el primer arranque).
- Se crea el admin si no existe.

## 7. Base de datos (PostgreSQL, migraciones con Alembic)
- `users(id, email UNIQUE, password_hash, role, created_at)`
- `training_files(id, filename, stored_path, rows, rows_inserted, rows_rejected, uploaded_by FK users, status, created_at)` — registro de cada carga de CSV (`status`: `loaded` o `failed`).
- `funding_rounds(id, round_id UNIQUE NULL, announced_date, year, company_name, company_domain, country, city, sector, founded_year, stage, amount_usd, amount_status, lead_investor, investor_count, news_mentions, news_source, news_url, fliar_profile_url, training_file_id FK NULL, created_at)` — las columnas del CSV **sin `news_excerpt`**; `training_file_id` NULL = CSV original. Índice en `(company_domain, announced_date)` para el historial.
- `model_versions(id, version INT UNIQUE, artifact_path, is_active BOOL, mae_log, r2, rmse_log, mdape, coverage, n_train, n_test, params JSONB, training_file_id FK NULL, created_at)` — `training_file_id` = última carga incluida al entrenar (NULL para la versión 1).
- `predictions(id, user_id FK, model_version_id FK, input JSONB, estimated_amount_usd, estimated_log10, created_at)`

Generar `db/schema.sql` y el diagrama ER (dbdiagram.io o mermaid).

## 8. Seguridad y configuración
Variables (`.env`, nunca en git): `DB_PASSWORD`, `DATABASE_URL`, `JWT_SECRET`, `JWT_ALGORITHM=HS256`, `JWT_EXPIRE_MINUTES`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ARTIFACTS_DIR`, `UPLOADS_DIR`, `CORS_ORIGINS`, `SEED_DATASET_PATH` (CSV original que se carga al primer arranque; vacío = no cargar).
Contraseñas con bcrypt. Crear el admin al arrancar si no existe.
**`joblib.load` solo sobre artefactos propios del proyecto** (`modelo_rondas.joblib` del notebook y los que genera `/retrain` en `ARTIFACTS_DIR`), nunca sobre archivos subidos por usuarios (deserialización insegura).
Versiones fijadas en `requirements.txt`, iguales a las del notebook: **scikit-learn `1.6.1`**, **pandas `2.2.3`** y **numpy `2.4.6`**. No cambiarlas sin regenerar el modelo (`TargetEncoder` requiere scikit-learn ≥ 1.3).

## 9. Pruebas mínimas (pytest)
Casos de uso con repositorios falsos; login y permisos (403 a `user` en `/retrain`); `/predict` con campos mínimos y con nulos; categoría desconocida no rompe; historial filtrado por usuario; subida de CSV inválido devuelve 422.
Además: el entrenamiento con el CSV original reproduce las métricas de la sección 1 (prueba lenta; se salta si el CSV no está en `data/`), y los repositorios se prueban contra PostgreSQL real con `TEST_DATABASE_URL`.

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
1. `domain` + `ports`. ✅
2. `sklearn_predictor.py` con el modelo base. ✅ (hay que adaptarlo al modelo de 14 variables)
3. Casos de uso con pruebas. ✅ (hay que adaptarlos al flujo con base de datos)
4. PostgreSQL + Alembic, con `funding_rounds`.
5. Modelo de 14 variables: dominio, predictor y contrato de `/predict`.
6. `cleaning.py`, `features.py`, `sklearn_trainer.py` (sección 4; reproducir R² 0,615).
7. Carga de CSV a `funding_rounds` y reentrenamiento desde la base de datos.
8. FastAPI, JWT y roles.
9. Docker Compose, con la demo del flujo del docente: subir datos → reentrenar → subir más → reentrenar.
10. `openapi.json`, README y `release/0.1.0`.

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
Nulos en el original: `country` 41, `city` 238, `sector` 158, `amount_usd` 3.457, `lead_investor` 3.791, `news_source`/`news_url` 8.659, `news_excerpt` 288, `fliar_profile_url` 42. `round_id` es único.

**Obligatorias en un CSV subido:** `announced_date`, `year`, `company_domain`, `founded_year`, `stage`, `sector`, `country`, `city`, `lead_investor`, `investor_count`, `news_mentions`, `news_source`, `amount_usd`, `amount_status`.
- `company_domain` es obligatoria: con ella se calcula el historial de cada empresa (`n_prev_rounds`, `prev_log_amount`).
- Opcionales (no entran al modelo): `round_id`, `company_name`, `news_url`, `fliar_profile_url`. Si viene `round_id` repetido, la fila se ignora.
- **NUNCA usar `news_excerpt` como variable**: si viene en el CSV, se descarta y no se guarda (ver sección 2).
