# Startup Funding API

API REST que estima el monto en USD de una ronda de financiación de una startup.
Trabajo final de la materia de IA. El cliente web vive en otro repositorio; este solo expone la API.

- **Modelo:** `HistGradientBoostingRegressor` dentro de un `Pipeline` de scikit-learn, entrenado sobre `log10(amount_usd)` con 14 variables. Viene del notebook `notebooks/Startups_Monto_CRISP_ML.ipynb`.
- **Métricas en test (3.190 filas):** R² 0,615 · MAE_log 0,350 · RMSE_log 0,495 · MdAPE 52,9 % · cobertura ±50 % del 47,3 %.
- **Datos:** dataset de Kaggle [`fliarbi/global-startup-funding-rounds-2019-2026`](https://www.kaggle.com/datasets/fliarbi/global-startup-funding-rounds-2019-2026), archivo `global_startup_funding_sample.csv` (20.227 filas). Licencia **CC BY-NC**, solo para uso académico. Por eso el CSV **no** está en el repositorio.

## Qué hace la API

| Requisito | Endpoint | Rol |
|---|---|---|
| Enviar datos al modelo y obtener la predicción | `POST /predict` | user, admin |
| Historial de predicciones: datos enviados y resultado | `GET /predictions?page=1&size=20` | user (solo las suyas), admin (todas) |
| Métricas de rendimiento del modelo | `GET /metrics` | user, admin |
| Reentrenar el modelo con información actualizada | `POST /retrain` | admin |
| Enviar el archivo de entrenamiento (CSV) | `POST /training-files` | admin |
| Registro e inicio de sesión (JWT) | `POST /auth/register`, `POST /auth/login` | público |
| Estado del servicio | `GET /health` | público |

El contrato completo está en [`docs/openapi.json`](docs/openapi.json). Con la API corriendo, también se puede ver y probar en `/docs` (Swagger).

**Errores:** siempre en JSON con la forma `{"detail": "..."}`.

| Código | Cuándo |
|---|---|
| 401 | sin token o con un token inválido |
| 403 | el rol no tiene permiso |
| 409 | email ya registrado, o ya hay un reentrenamiento en curso |
| 422 | entrada inválida, CSV inválido o datos insuficientes para la partición estratificada |
| 503 | no hay un modelo activo |

## Puesta en marcha con Docker (recomendada)

Requisitos: Docker con Compose v2.

1. Copia el CSV de Kaggle en `data/global_startup_funding_sample.csv`.
2. Crea el `.env` a partir del ejemplo y cambia los secretos:
   ```bash
   cp .env.example .env
   # DB_PASSWORD, JWT_SECRET (al menos 32 caracteres: openssl rand -hex 32) y ADMIN_PASSWORD
   ```
3. Levanta la base de datos y la API:
   ```bash
   docker compose up -d --build
   ```
4. Abre <http://localhost:8000/docs>. Si el puerto 8000 está ocupado, usa `API_PORT=8001 docker compose up -d`.

**Qué pasa en el primer arranque:**
1. Se aplican las migraciones de Alembic.
2. Si la tabla `funding_rounds` está vacía, se carga el CSV original.
3. `artifacts/modelo_rondas.joblib` se registra como **versión 1 activa**, así que la API predice desde el inicio.
4. Se crea el admin del `.env`.

Cada paso es idempotente: en los arranques siguientes no repite nada.

### Probar desde Swagger

1. `POST /auth/login` con `ADMIN_EMAIL` y `ADMIN_PASSWORD`. Copia el `access_token`.
2. Pulsa **Authorize** y pega el token.
3. `POST /predict` con el cuerpo de ejemplo:
   ```json
   {
     "year": 2025, "founded_year": 2021, "stage": "Seed", "sector": "Fintech",
     "country": "United States", "city": "New York",
     "investor_count": 3, "lead_investor": "Y Combinator",
     "news_mentions": 2, "news_source": "TechCrunch",
     "company_domain": "acme.io", "prev_rounds": 1, "prev_amount_usd": 1500000
   }
   ```
   Son obligatorios `year`, `founded_year`, `stage`, `sector` y `country`. La respuesta trae:
   - `estimated_amount_usd`: el monto estimado.
   - `estimated_log10`: el mismo valor en log10.
   - `range_usd`: el estimado × / ÷ 1,5.
   - `model_version`: la versión que predijo.
   - `warning`: un aviso cuando `stage` es `Undisclosed`.

## Demo del flujo del docente

La secuencia es: subir datos → reentrenar → subir más datos → reentrenar.

```bash
python scripts/teacher_demo.py prepare            # divide el CSV: 80 % semilla y dos lotes de 10 %
docker compose down -v                            # ¡borra la base! La semilla solo se carga en una base vacía
SEED_DATASET_PATH=./data/demo/seed.csv docker compose up -d --build
python scripts/teacher_demo.py run                # --url http://localhost:8001 si cambiaste API_PORT
```

`prepare` divide el CSV en el orden del archivo. Así cada lote trae rondas reales que la base todavía no tiene. Resultado obtenido:

| Paso | Resultado |
|---|---|
| Inicio | v1 activa · R² 0,6153 · MAE_log 0,3504 |
| Subir el lote 1 (2.023 filas) → reentrenar | v2 **inactiva**: MAE_log 0,3644 > 0,3504 + 0,01; la API sigue con la v1 |
| Subir el lote 2 (2.022 filas) → reentrenar | v3 **activa** · R² 0,6153 · MAE_log 0,3504, igual que el notebook |

### Cómo funciona `/retrain`

- Cada reentrenamiento usa **todas** las filas de `funding_rounds` para calcular las variables de historial, como `n_prev_rounds`, `prev_log_amount` y `lead_freq`.
- La **partición 80/20 estratificada nueva** (`random_state=42`) solo incluye las rondas usables: `amount_status == "verified"` y `amount_usd` de al menos 1.000 USD. Una ronda `not_disclosed` o `failed_check` cuenta para el historial de su empresa, pero no entra ni en el entrenamiento ni en el test.
- Entrena con los hiperparámetros ganadores del notebook, sin repetir la búsqueda.
- La versión nueva se activa solo si `MAE_log_nuevo <= MAE_log_activo + 0.01`, comparando con el MAE **guardado** de la versión activa. Si no, se guarda inactiva y se avisa en `message`.
- **Límite:** cada versión se mide sobre su propio test. Una diferencia pequeña de MAE puede deberse al sorteo de la partición y no solo al modelo.
- Un reentrenamiento que falla, por cualquier motivo, no toca el modelo activo.

### Formato del CSV de entrenamiento

- **Columnas obligatorias:** `announced_date`, `year`, `company_domain`, `founded_year`, `stage`, `sector`, `country`, `city`, `lead_investor`, `investor_count`, `news_mentions`, `news_source`, `amount_usd`, `amount_status`.
- **Opcionales:** `round_id`, `company_name`, `news_url`, `fliar_profile_url`.
- **Archivo:** extensión `.csv` y como máximo 20 MB.
- **Si `year` no coincide con el año de `announced_date`:** se rechaza el archivo completo con 422.
- **Filas que se rechazan:** se cuentan en `rows_rejected`.
  - `founded_year` nulo, o `company_age` negativa (`founded_year` posterior a `year`);
  - `company_domain`, `stage` o `amount_status` vacíos.
- **Filas que se ignoran:** las que traen un `round_id` que ya existe.
- **`news_excerpt`:** contiene el monto escrito (fuga de datos), así que nunca se usa para entrenar y no se guarda en la base de datos. Sí queda en la copia del CSV original que se guarda en `UPLOADS_DIR`, porque esa copia se guarda tal cual. Si no quieres que quede guardado, quita la columna antes de subir el archivo.

## Desarrollo local (sin Docker)

Requisitos: Python 3.12 y PostgreSQL 16.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # DATABASE_URL debe apuntar a tu PostgreSQL
alembic upgrade head
uvicorn app.main:app --reload
```

### Variables de entorno

| Variable | Descripción |
|---|---|
| `DB_PASSWORD` | contraseña de PostgreSQL (la usa docker compose) |
| `DATABASE_URL` | `postgresql+psycopg://usuario:clave@host:5432/base`; en Docker se calcula sola |
| `JWT_SECRET` | secreto para firmar los tokens; **al menos 32 caracteres** |
| `JWT_ALGORITHM` / `JWT_EXPIRE_MINUTES` | `HS256` / `60` |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | admin que se crea al arrancar si no existe |
| `ARTIFACTS_DIR` | carpeta de los modelos (`./artifacts`) |
| `UPLOADS_DIR` | copia de los CSV subidos (`./data/uploads`) |
| `CORS_ORIGINS` | orígenes permitidos, separados por comas |
| `SEED_DATASET_PATH` | CSV que se carga en una base vacía al arrancar; vacío = no cargar |
| `API_PORT` | puerto del host en docker compose (8000 por defecto) |

`.env` nunca va en git.

## Pruebas

```bash
pytest -q
```

- Las pruebas de repositorios usan un **PostgreSQL real**; se saltan si no hay `TEST_DATABASE_URL`:
  ```bash
  docker run -d --rm --name fundcast-test-db -e POSTGRES_PASSWORD=test \
    -e POSTGRES_DB=fundcast_test -p 55432:5432 postgres:16
  TEST_DATABASE_URL=postgresql+psycopg://postgres:test@127.0.0.1:55432/fundcast_test pytest -q
  ```
- La prueba lenta que reproduce las métricas del notebook se salta si el CSV no está en `data/`.
- `tests/integration/test_openapi_export.py` falla si `docs/openapi.json` quedó desactualizado. Para regenerarlo: `python -m scripts.export_openapi`.

## Arquitectura

Es una arquitectura hexagonal (puertos y adaptadores):

```
app/
├── domain/            entidades y reglas de negocio (solo biblioteca estándar)
├── application/
│   ├── ports/         interfaces: predictor, trainer, repositorios, storage, seguridad
│   └── use_cases/     predict_round, retrain_model, upload_training_file, login, ...
├── adapters/
│   ├── inbound/api/   FastAPI: routers, esquemas, JWT y roles
│   └── outbound/      ml/ (scikit-learn), postgres/ (SQLAlchemy), security/, storage/
├── config.py          settings desde variables de entorno
└── main.py            composición: conecta puertos y adaptadores
```

**Regla de dependencias:**
- `domain` no importa nada.
- `application` solo importa `domain` y los puertos.
- Los routers solo llaman a casos de uso.
- scikit-learn, pandas, joblib y SQLAlchemy solo aparecen en `adapters/outbound`.

`tests/unit/test_architecture.py` hace cumplir estas reglas.

**Otros documentos:**
- Base de datos: [`db/schema.sql`](db/schema.sql) y el [diagrama ER](db/er-diagram.md).
- Seguridad:
  - Las contraseñas se guardan con bcrypt.
  - `joblib.load` solo se aplica a los artefactos propios del proyecto, nunca a archivos subidos.
  - Los archivos subidos se guardan con un nombre generado por el servidor.

## Flujo de trabajo

- **Ramas (Git Flow):** `main` para las entregas, `develop` para la integración y una rama `feature/*` por paso.
- **Commits:** siguen Conventional Commits.
- **Pull Requests:** cada uno se revisa con IA (CodeRabbit, comentando `@coderabbitai review`). Los hallazgos válidos se corrigen con pruebas y se responden en el hilo.
