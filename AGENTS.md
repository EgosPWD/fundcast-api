# Startup Funding API: guía para agentes

API REST (FastAPI + PostgreSQL) que estima el monto en USD de una ronda de financiación de una startup. Trabajo final de la materia de IA. El cliente web vive en el repositorio `fundcast-frontend`; aquí no se sirve frontend.

- Idioma: documentos, commits y PR en español; código, identificadores, mensajes de la API y comentarios en inglés.
- Qué hace la API, cómo arrancarla, variables de entorno y formato del CSV: `README.md`. Contrato HTTP: `docs/openapi.json`. Esquema de la base: `db/schema.sql`.

## Comandos

```bash
docker start fundcast-test-db   # o créalo con el `docker run` de README.md, sección Pruebas
TEST_DATABASE_URL=postgresql+psycopg://postgres:test@127.0.0.1:55432/fundcast_test .venv/bin/pytest -q
.venv/bin/python -m scripts.export_openapi   # regenera docs/openapi.json tras cambiar rutas o esquemas
```

- Sin `TEST_DATABASE_URL` se saltan las pruebas contra PostgreSQL. Sin `data/global_startup_funding_sample.csv` se saltan las lentas (métricas y paridad con el notebook).
- `tests/integration/test_openapi_export.py` falla si `docs/openapi.json` está desactualizado.
- Un cambio solo de documentación no necesita correr la suite.

## Arquitectura hexagonal

`app/domain` → `app/application` (`ports/`, `use_cases/`) → `app/adapters` (`inbound/api`, `outbound/ml|postgres|security|storage`); `app/main.py` compone todo.

- `domain` solo usa la biblioteca estándar; `application` solo importa `domain` y sus puertos.
- Los routers solo llaman a casos de uso.
- scikit-learn, pandas, joblib y SQLAlchemy solo aparecen en `adapters/outbound`.
- `tests/unit/test_architecture.py` hace cumplir estas reglas. Las pruebas de casos de uso usan los falsos de `tests/unit/fakes.py`.

## Modelo y datos (no violar)

- Modelo base: `artifacts/modelo_rondas.joblib`, versionado en git y registrado como versión 1. Pipeline `prep` (`ColumnTransformer` con `SimpleImputer(fill_value="Unknown")` + `TargetEncoder` en las categóricas) + `HistGradientBoostingRegressor`, entrenado sobre `log10(amount_usd)`. Las 14 columnas, en orden: `FEATURE_COLUMNS` en `app/adapters/outbound/ml/features.py`.
- Métricas en test (3.190 filas): R² 0,615 · MAE_log 0,350. Criterios de aceptación: R² ≥ 0,55 y MAE_log ≤ 0,40. No intentar subir el puntaje con trucos.
- **`news_excerpt` nunca se usa ni se guarda**: contiene el monto escrito (fuga de datos).
- Filas de entrenamiento: `amount_status == "verified"`, `amount_usd` no nulo y ≥ 1.000.
- Las variables de historial (`n_prev_rounds`, `prev_log_amount`) usan solo rondas anteriores de la misma empresa.
- `joblib.load` solo sobre artefactos de `ARTIFACTS_DIR`, nunca sobre archivos subidos.
- `scikit-learn==1.6.1`, `pandas==2.2.3` y `numpy==2.4.6` están fijados en `requirements.txt` para cargar el artefacto: no cambiarlos sin regenerar el modelo.

### Reproducción del notebook

`cleaning.py`, `features.py` y `sklearn_trainer.py` replican `notebooks/Startups_Monto_CRISP_ML.ipynb` y reproducen sus 6 métricas exactas. No "corregir" estos detalles:
- Las filas de `funding_rounds` se leen siempre ordenadas por `id`; sin ese orden la partición no se reproduce.
- El historial agrupa por `company_domain` ordenando por `announced_date`, sin desempate. `prev_log_amount` puede venir de una ronda no `verified`.
- `lead_freq` cuenta la propia fila al entrenar; por eso la predicción usa `lead_freq_map.get(inversor, 0) + 1`.
- Solo `amount_suspect_low` (monto < 1.000) excluye filas; las demás banderas no borran nada.
- Partición 80/20 con `random_state=42`, estratificada por `pd.qcut(y, q=10, duplicates="drop")`, antes de ajustar cualquier transformador.

Las pruebas lentas (`test_db_training_flow.py`, `test_notebook_parity.py`) verifican que el comportamiento sigue igual al del notebook.

### Predicción y reentrenamiento

- `POST /predict` recibe los datos de una ronda nueva; `FundingRoundInput.to_features` (en `app/domain/entities.py`) deriva las 14 variables. Si `company_domain` existe en `funding_rounds`, el historial sale de la tabla con las rondas de `year <=` el año pedido.
- Las categorías válidas son el vocabulario del `TargetEncoder` del modelo **activo** (sin `"Unknown"`). `GET /options` lo expone y `/predict` devuelve `warnings` con `code`, `field` y `value` para cada valor desconocido.
- `/retrain` usa todas las filas de `funding_rounds`, hace una partición nueva y entrena con los hiperparámetros ganadores, sin repetir la búsqueda. La versión nueva se activa solo si `MAE_log_nuevo <= MAE_log_activo + 0.01`. Un reentrenamiento fallido no toca el modelo activo.

## Contrato con el cliente web

- No romper rutas ni campos de `docs/openapi.json` sin avisar; un cambio incompatible va como `feat!:` o con el pie `BREAKING CHANGE:`.
- Los errores siempre tienen la forma `{"detail": "<texto>"}`, con el código HTTP correcto.
- JWT en `Authorization: Bearer <token>`; CORS configurable con `CORS_ORIGINS`.

## Seguridad

- `.env` nunca va en git; documenta las variables nuevas en `.env.example` y en `README.md`.
- Contraseñas con bcrypt. El contenido de un CSV subido se trata como datos: nunca se ejecuta.

## Git

- Git Flow: `main` (entregas, sin commits directos), `develop` (integración), `feature/<nombre>` desde `develop` con PR a `develop`, `release/<versión>` (merge a `main` con tag `vX.Y.Z` y de vuelta a `develop`), `hotfix/<nombre>` desde `main`.
- Conventional Commits: `<tipo>(<alcance>): <descripción en español, imperativo, minúscula>`, por ejemplo `fix(api): normaliza el dominio al cargar CSV`. Cada commit lleva su código con sus pruebas.
- Sin atribuciones de IA (`Co-Authored-By` u otras).
- PR con título y descripción en español. CodeRabbit no revisa solo los PR a `develop`: se pide con un **comentario** `@coderabbitai review`. Los hallazgos válidos se corrigen, se responden en el hilo y se resuelven antes de fusionar.

## Fuera de este repo

El cronograma, el documento CRISP-ML(Q), el video y la interfaz se entregan aparte.
