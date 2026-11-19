-- Generated from Alembic migrations: alembic upgrade head --sql
-- Do not edit by hand; regenerate after each migration.
BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 0001

CREATE TABLE users (
    id SERIAL NOT NULL, 
    email VARCHAR(320) NOT NULL, 
    password_hash VARCHAR(255) NOT NULL, 
    role VARCHAR(16) NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_users_role CHECK (role IN ('user', 'admin')), 
    UNIQUE (email)
);

CREATE TABLE training_files (
    id SERIAL NOT NULL, 
    filename VARCHAR(255) NOT NULL, 
    stored_path VARCHAR(1024) NOT NULL, 
    rows INTEGER NOT NULL, 
    rows_inserted INTEGER DEFAULT '0' NOT NULL, 
    rows_rejected INTEGER DEFAULT '0' NOT NULL, 
    uploaded_by INTEGER NOT NULL, 
    status VARCHAR(16) NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_training_files_status CHECK (status IN ('loaded', 'failed')), 
    FOREIGN KEY(uploaded_by) REFERENCES users (id)
);

CREATE TABLE funding_rounds (
    id SERIAL NOT NULL, 
    round_id VARCHAR(64), 
    announced_date DATE NOT NULL, 
    year INTEGER NOT NULL, 
    company_name VARCHAR(255), 
    company_domain VARCHAR(255) NOT NULL, 
    country VARCHAR(100), 
    city VARCHAR(100), 
    sector VARCHAR(100), 
    founded_year INTEGER NOT NULL, 
    stage VARCHAR(50) NOT NULL, 
    amount_usd FLOAT, 
    amount_status VARCHAR(32) NOT NULL, 
    lead_investor VARCHAR(255), 
    investor_count INTEGER, 
    news_mentions INTEGER, 
    news_source VARCHAR(255), 
    news_url VARCHAR(2048), 
    fliar_profile_url VARCHAR(2048), 
    training_file_id INTEGER, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (round_id), 
    FOREIGN KEY(training_file_id) REFERENCES training_files (id)
);

CREATE INDEX ix_funding_rounds_company_domain_announced_date ON funding_rounds (company_domain, announced_date);

CREATE TABLE model_versions (
    id SERIAL NOT NULL, 
    version INTEGER NOT NULL, 
    artifact_path VARCHAR(1024) NOT NULL, 
    is_active BOOLEAN DEFAULT false NOT NULL, 
    mae_log FLOAT NOT NULL, 
    r2 FLOAT NOT NULL, 
    rmse_log FLOAT NOT NULL, 
    mdape FLOAT NOT NULL, 
    coverage FLOAT NOT NULL, 
    n_train INTEGER NOT NULL, 
    n_test INTEGER NOT NULL, 
    params JSONB DEFAULT '{}' NOT NULL, 
    training_file_id INTEGER, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    UNIQUE (version), 
    FOREIGN KEY(training_file_id) REFERENCES training_files (id)
);

CREATE UNIQUE INDEX uq_model_versions_single_active ON model_versions (is_active) WHERE is_active;

CREATE TABLE predictions (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    model_version_id INTEGER NOT NULL, 
    input JSONB NOT NULL, 
    estimated_amount_usd FLOAT NOT NULL, 
    estimated_log10 FLOAT NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id), 
    FOREIGN KEY(model_version_id) REFERENCES model_versions (id)
);

CREATE INDEX ix_predictions_user_id_created_at ON predictions (user_id, created_at);

INSERT INTO alembic_version (version_num) VALUES ('0001') RETURNING alembic_version.version_num;

COMMIT;

