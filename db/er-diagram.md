# Diagrama entidad-relación

Generado a partir de `alembic/versions/0001_initial_schema.py`. GitHub renderiza el bloque mermaid.

`funding_rounds` guarda las columnas del CSV **sin `news_excerpt`**: esa columna contiene el monto escrito (fuga de datos) y nunca se almacena.

```mermaid
erDiagram
    users ||--o{ predictions : "hace"
    users ||--o{ training_files : "sube"
    model_versions ||--o{ predictions : "genera"
    training_files |o--o{ model_versions : "última carga incluida"
    training_files |o--o{ funding_rounds : "carga"

    users {
        int id PK
        varchar email UK
        varchar password_hash "bcrypt"
        varchar role "user | admin"
        timestamptz created_at
    }
    training_files {
        int id PK
        varchar filename
        varchar stored_path
        int rows "filas recibidas"
        int rows_inserted
        int rows_rejected
        int uploaded_by FK
        varchar status "loaded | failed"
        timestamptz created_at
    }
    funding_rounds {
        int id PK
        varchar round_id UK "nullable; repetidos se ignoran"
        date announced_date
        int year
        varchar company_name
        varchar company_domain "clave del historial"
        varchar country
        varchar city
        varchar sector
        int founded_year
        varchar stage
        float amount_usd
        varchar amount_status
        varchar lead_investor
        int investor_count
        int news_mentions
        varchar news_source
        varchar news_url
        varchar fliar_profile_url
        int training_file_id FK "null = CSV original"
        timestamptz created_at
    }
    model_versions {
        int id PK
        int version UK
        varchar artifact_path
        bool is_active "a lo sumo una activa (índice único parcial)"
        float mae_log
        float r2
        float rmse_log
        float mdape
        float coverage
        int n_train
        int n_test
        jsonb params
        int training_file_id FK "última carga incluida; null para la v1"
        timestamptz created_at
    }
    predictions {
        int id PK
        int user_id FK
        int model_version_id FK
        jsonb input "cuerpo enviado a POST /predict"
        float estimated_amount_usd
        float estimated_log10
        timestamptz created_at
    }
```
