# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the Weather Intelligence application.
# MAGIC
# MAGIC Pipeline:
# MAGIC
# MAGIC ```text
# MAGIC National Weather Service API
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/sync
# MAGIC             |
# MAGIC             v
# MAGIC        weather_news
# MAGIC             |
# MAGIC             v
# MAGIC ingest_weather_embeddings
# MAGIC             |
# MAGIC       chunk narrative_text
# MAGIC             |
# MAGIC     all-MiniLM-L6-v2
# MAGIC             |
# MAGIC             v
# MAGIC     weather_embeddings
# MAGIC             |
# MAGIC       pgvector / HNSW
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/search
# MAGIC ```
# MAGIC
# MAGIC The notebook:
# MAGIC
# MAGIC 1. Reads weather documents from `weather_news`.
# MAGIC 2. Finds documents that do not yet have embeddings.
# MAGIC 3. Splits `narrative_text` into overlapping chunks.
# MAGIC 4. Generates 384-dimensional embeddings using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2`.
# MAGIC 5. Writes embeddings directly to Lakebase.
# MAGIC 6. Stores embeddings in `weather_embeddings.embedding` as
# MAGIC    PostgreSQL `VECTOR(384)`.
# MAGIC 7. Validates pgvector and cosine similarity search.
# MAGIC
# MAGIC IMPORTANT:
# MAGIC
# MAGIC This notebook intentionally does NOT use:
# MAGIC
# MAGIC - psycopg2
# MAGIC - psycopg2-binary
# MAGIC - Spark JDBC for writes
# MAGIC - the legacy Lakebase Database Instance API
# MAGIC - a manually stored PostgreSQL password
# MAGIC
# MAGIC Database connections use:
# MAGIC
# MAGIC ```text
# MAGIC WorkspaceClient
# MAGIC       |
# MAGIC       v
# MAGIC postgres.generate_database_credential()
# MAGIC       |
# MAGIC       v
# MAGIC short-lived Lakebase credential
# MAGIC       |
# MAGIC       v
# MAGIC pg8000
# MAGIC ```
=======
# MAGIC 1. Reads weather documents harvested from the National Weather Service
# MAGIC    (NWS) API and stored in `weather_news`.
# MAGIC 2. Finds documents that do not yet have embeddings.
# MAGIC 3. Splits each document's `narrative_text` into overlapping chunks.
# MAGIC 4. Embeds each chunk using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
# MAGIC 5. Writes embeddings directly to Lakebase using `pg8000`.
# MAGIC 6. Stores the embeddings in a PostgreSQL `VECTOR(384)` column.
# MAGIC 7. Verifies the pgvector column and performs a cosine-similarity test.
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint is responsible for
# MAGIC harvesting NWS data and populating `weather_news`.
# MAGIC
# MAGIC The resulting pipeline is:
# MAGIC
# MAGIC ```text
# MAGIC NWS API
# MAGIC    |
# MAGIC    v
# MAGIC /weather/sync
# MAGIC    |
# MAGIC    v
# MAGIC weather_news
# MAGIC    |
# MAGIC    v
# MAGIC this notebook
# MAGIC    |
# MAGIC    +--> chunk narrative_text
# MAGIC    |
# MAGIC    +--> all-MiniLM-L6-v2
# MAGIC    |
# MAGIC    v
# MAGIC weather_embeddings
# MAGIC    |
# MAGIC    v
# MAGIC pgvector / HNSW
# MAGIC    |
# MAGIC    v
# MAGIC /weather/search
# MAGIC ```
# MAGIC
# MAGIC ## Important Serverless Compatibility Note
# MAGIC
# MAGIC This notebook intentionally does NOT use:
# MAGIC
# MAGIC - `psycopg2`
# MAGIC - `psycopg2-binary`
# MAGIC - `spark.write.jdbc`
# MAGIC
# MAGIC `psycopg2-binary` contains native C extensions that can cause
# MAGIC `SIGABRT 134` kernel crashes on Databricks Serverless compute.
# MAGIC
# MAGIC Instead, this notebook uses **pg8000**, a pure-Python PostgreSQL driver.
# MAGIC
# MAGIC Embeddings are also written directly to the PostgreSQL `VECTOR` column,
# MAGIC so no post-processing cast from arrays to vectors is required.

# COMMAND ----------

# DBTITLE 1,Setup Instructions
# MAGIC %md
# MAGIC ## ✅ Lakebase Instance Status
# MAGIC
# MAGIC This notebook connects to your Lakebase Postgres instance:
# MAGIC
# MAGIC - **Project**: `dataexpert-student`
# MAGIC - **Branch**: `production` (READY)
# MAGIC - **Endpoint**: `primary` (ACTIVE)  
# MAGIC - **Database**: `databricks_postgres`
# MAGIC - **PostgreSQL**: 17.10
# MAGIC - **pgvector**: 0.8.0 ✓
# MAGIC
# MAGIC **Connection verified and working!**
# MAGIC
# MAGIC ### Required Setup
# MAGIC
# MAGIC #### 1. Create Required Tables
# MAGIC
# MAGIC Run these SQL scripts in your Lakebase database (in order):
# MAGIC
# MAGIC ```bash
# MAGIC # From the project root:
# MAGIC psql "$(cat lakebase-url-from-secret)" -f sql/01_setup_weather_news_table.sql
# MAGIC psql "$(cat lakebase-url-from-secret)" -f sql/02_setup_weather_embeddings.sql
# MAGIC ```
# MAGIC
# MAGIC Or connect via pgAdmin/DBeaver and run the SQL files manually.
# MAGIC
# MAGIC ### 3. Sync Weather Data
# MAGIC
# MAGIC Before running this notebook, call the Flask API to fetch and store weather documents:
# MAGIC
# MAGIC ```bash
# MAGIC curl -X POST http://your-app-url/weather/sync \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{
# MAGIC     "locations": ["Chicago, IL", "Austin, TX"],
# MAGIC     "limit": 50
# MAGIC   }'
# MAGIC ```
# MAGIC
# MAGIC ### 4. Once Setup is Complete
# MAGIC
# MAGIC Run all cells in order. The notebook will:
# MAGIC 1. Connect to Lakebase
# MAGIC 2. Find unembedded weather documents  
# MAGIC 3. Chunk the narrative text
# MAGIC 4. Generate embeddings using sentence-transformers
# MAGIC 5. Write embeddings back to Lakebase with pgvector
# MAGIC
# MAGIC The notebook is idempotent - safe to re-run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC `pg8000` is used instead of `psycopg2-binary`.
# MAGIC
# MAGIC `pg8000` is a pure-Python PostgreSQL driver and avoids the native C
# MAGIC extension issues that can cause SIGABRT kernel crashes on Databricks
# MAGIC Serverless compute.
# MAGIC
# MAGIC `databricks-sdk` is used to generate a fresh Lakebase database credential.
# MAGIC
# MAGIC `sentence-transformers` generates the weather embeddings.
# MAGIC - **pg8000**: Pure-Python PostgreSQL/Lakebase driver
# MAGIC - **sentence-transformers**: Embedding model library
# MAGIC
# MAGIC The NWS API is called by the Flask application's `weather_client.py`.
# MAGIC This notebook only processes documents already stored in Lakebase.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q pg8000 sentence-transformers sqlalchemy requests
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup paths and import lakebase
import sys
import os

# Add parent directory to path so we can import project modules
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the lakebase connection helper
import lakebase

print(f"✓ Project root: {project_root}")
print(f"✓ lakebase module loaded from: {lakebase.__file__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
<<<<<<< Updated upstream
# MAGIC The notebook can be run manually or scheduled as a Databricks Job.
# MAGIC
# MAGIC The Lakebase endpoint must be the **Lakebase PostgreSQL endpoint resource
# MAGIC name**, not the old Database Instance name.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```text
# MAGIC projects/dataexpert-student/branches/production/endpoints/primary
# MAGIC ```
# MAGIC
# MAGIC The exact endpoint for your project should be confirmed before running
# MAGIC the notebook.
=======
# MAGIC These widgets allow the notebook to be run manually or scheduled as a
# MAGIC Databricks Job without modifying the source code.
>>>>>>> Stashed changes

# COMMAND ----------

dbutils.widgets.text(
    "lakebase_endpoint_name",
    "projects/dataexpert-student/branches/production/endpoints/primary",
    "Lakebase PostgreSQL endpoint"
)

dbutils.widgets.text(
    "weather_table_name",
    "weather_news",
    "Source table (weather documents)"
)

dbutils.widgets.text(
    "embeddings_table_name",
    "weather_embeddings",
    "Destination table (chunk embeddings)"
)

dbutils.widgets.text(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Embedding model"
)

dbutils.widgets.text(
    "chunk_size",
    "800",
    "Chunk size (characters)"
)

dbutils.widgets.text(
    "chunk_overlap",
    "100",
    "Chunk overlap (characters)"
)

dbutils.widgets.text(
    "batch_size",
    "50",
    "Embedding/write batch size"
)

<<<<<<< Updated upstream
LAKEBASE_ENDPOINT_NAME = dbutils.widgets.get(
    "lakebase_endpoint_name"
)

WEATHER_TABLE_NAME = dbutils.widgets.get(
    "weather_table_name"
)

EMBEDDINGS_TABLE_NAME = dbutils.widgets.get(
    "embeddings_table_name"
=======
dbutils.widgets.text(
    "database_secret_scope",
    "database",
    "Lakebase secret scope"
)

dbutils.widgets.text(
    "database_secret_key",
    "lakebase-url",
    "Lakebase secret key"
>>>>>>> Stashed changes
)

EMBEDDING_MODEL_NAME = dbutils.widgets.get(
    "embedding_model"
)

CHUNK_SIZE = int(
    dbutils.widgets.get("chunk_size")
)

<<<<<<< Updated upstream
CHUNK_OVERLAP = int(
    dbutils.widgets.get("chunk_overlap")
)

BATCH_SIZE = int(
    dbutils.widgets.get("batch_size")
)

# Assignment requires all-MiniLM-L6-v2.
if EMBEDDING_MODEL_NAME != "sentence-transformers/all-MiniLM-L6-v2":
=======
DATABASE_SECRET_SCOPE = dbutils.widgets.get("database_secret_scope")
DATABASE_SECRET_KEY = dbutils.widgets.get("database_secret_key")

# The assignment specifies all-MiniLM-L6-v2 / 384 dimensions.
if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    EMBEDDING_DIM = 384
else:
>>>>>>> Stashed changes
    raise ValueError(
        f"Unsupported embedding model: {EMBEDDING_MODEL_NAME!r}. "
        "This application uses "
        "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."
    )

<<<<<<< Updated upstream
EMBEDDING_DIM = 384

if CHUNK_SIZE <= 0:
    raise ValueError("chunk_size must be greater than zero")

if CHUNK_OVERLAP < 0:
    raise ValueError("chunk_overlap cannot be negative")

=======
>>>>>>> Stashed changes
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "chunk_overlap must be smaller than chunk_size"
    )

if CHUNK_SIZE <= 0:
    raise ValueError("chunk_size must be greater than zero")

if BATCH_SIZE <= 0:
    raise ValueError(
        "batch_size must be greater than zero"
    )

print("Configuration")
print("=" * 70)
print(f"Lakebase endpoint:      {LAKEBASE_ENDPOINT_NAME}")
print(f"Weather table:          {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")
<<<<<<< Updated upstream
print(f"Chunk size:             {CHUNK_SIZE}")
print(f"Chunk overlap:          {CHUNK_OVERLAP}")
print(f"Batch size:             {BATCH_SIZE}")
=======
print(f"Chunk size:              {CHUNK_SIZE}")
print(f"Chunk overlap:           {CHUNK_OVERLAP}")
print(f"Batch size:              {BATCH_SIZE}")
print(f"Secret scope:            {DATABASE_SECRET_SCOPE}")
print(f"Secret key:              {DATABASE_SECRET_KEY}")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
<<<<<<< Updated upstream
# MAGIC ## Import Libraries
# MAGIC
# MAGIC `pg8000` is the only PostgreSQL driver used by this notebook.

# COMMAND ----------

import os
import pg8000.native

from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Databricks Workspace Client
# MAGIC
# MAGIC The notebook uses the Databricks identity available to the compute
# MAGIC environment.
# MAGIC
# MAGIC No static Lakebase database password is stored in the notebook.
=======
# MAGIC ## Resolve Lakebase Connection
# MAGIC
# MAGIC The Lakebase connection URL is stored in the Databricks secret:
# MAGIC
# MAGIC ```text
# MAGIC scope = database
# MAGIC key   = lakebase-url
# MAGIC ```
# MAGIC
# MAGIC The value is the same Base64-encoded PostgreSQL URL used by the
# MAGIC application, for example:
# MAGIC
# MAGIC ```text
# MAGIC postgresql://role:password@host:5432/database?sslmode=require
# MAGIC ```
# MAGIC
# MAGIC We intentionally create the connection directly with `pg8000`.
# MAGIC
# MAGIC We do NOT import `lakebase.py` here because an implementation of
# MAGIC `lakebase.py` that imports `psycopg2-binary` can cause Serverless kernel
# MAGIC crashes.

# COMMAND ----------

import base64
import re
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

import pg8000.native

w = WorkspaceClient()

def get_lakebase_url() -> str:
    """Read and decode the Lakebase PostgreSQL URL from Databricks Secrets."""
    secret = w.secrets.get_secret(
        scope=DATABASE_SECRET_SCOPE,
        key=DATABASE_SECRET_KEY,
    )
>>>>>>> Stashed changes

    if not secret.value:
        raise ValueError(
            f"Secret {DATABASE_SECRET_SCOPE}/{DATABASE_SECRET_KEY} is empty."
        )

<<<<<<< Updated upstream
w = WorkspaceClient()

print("Databricks WorkspaceClient initialized successfully.")
=======
from urllib.parse import urlparse, quote_plus

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")

lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

if not parsed.hostname:
    raise ValueError("Lakebase URL does not contain a hostname.")

if not parsed.username:
    raise ValueError("Lakebase URL does not contain a username.")

if not parsed.password:
    raise ValueError("Lakebase URL does not contain a password.")

if not parsed.path or parsed.path == "/":
    raise ValueError("Lakebase URL does not contain a database name.")

DB_HOST = parsed.hostname
DB_PORT = parsed.port or 5432
DB_NAME = parsed.path.lstrip("/")
DB_USER = parsed.username
DB_PASSWORD = parsed.password

print(f"Lakebase host:     {DB_HOST}")
print(f"Lakebase port:     {DB_PORT}")
print(f"Lakebase database: {DB_NAME}")
print(f"Lakebase user:     {DB_USER}")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## PostgreSQL Identifier Validation
# MAGIC
<<<<<<< Updated upstream
# MAGIC Lakebase Autoscaling Projects use the PostgreSQL endpoint API.
# MAGIC
# MAGIC This is intentionally different from the legacy:
# MAGIC
# MAGIC ```python
# MAGIC w.database.generate_database_credential(
# MAGIC     instance_names=[...]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC We instead use:
# MAGIC
# MAGIC ```python
# MAGIC w.postgres.generate_database_credential(
# MAGIC     endpoint=...
# MAGIC )
# MAGIC ```

# COMMAND ----------

import re

endpoint_pattern = (
    r"^projects/[^/]+/branches/[^/]+/endpoints/[^/]+$"
)

if not re.match(
    endpoint_pattern,
    LAKEBASE_ENDPOINT_NAME
):
    raise ValueError(
        "LAKEBASE_ENDPOINT_NAME does not appear to be a "
        "Lakebase PostgreSQL endpoint resource name.\n\n"
        "Expected format:\n"
        "projects/<project>/branches/<branch>/endpoints/<endpoint>\n\n"
        f"Received:\n{LAKEBASE_ENDPOINT_NAME}"
    )

print(
    "Lakebase endpoint format validated:"
)
print(
    LAKEBASE_ENDPOINT_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Fresh Lakebase Credential
# MAGIC
# MAGIC Lakebase credentials are short-lived.
# MAGIC
# MAGIC The credential is generated when the notebook needs to connect instead
# MAGIC of relying on a potentially stale password stored in a secret.

# COMMAND ----------

def generate_lakebase_credential():
    """
    Generate a fresh short-lived Lakebase PostgreSQL credential.
    """

    credential = (
        w.postgres.generate_database_credential(
            endpoint=LAKEBASE_ENDPOINT_NAME
        )
    )

    return credential


# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse Lakebase Endpoint
# MAGIC
# MAGIC The PostgreSQL endpoint hostname is derived from the Lakebase endpoint
# MAGIC resource.
# MAGIC
# MAGIC The generated credential contains the authentication token.
# MAGIC
# MAGIC For the Weather Intelligence project the endpoint hostname follows the
# MAGIC Lakebase PostgreSQL format:
# MAGIC
# MAGIC ```text
# MAGIC ep-<endpoint>.<project>.<region>.cloud.databricks.com
# MAGIC ```
# MAGIC
# MAGIC The endpoint resource itself is authoritative for authentication.

# COMMAND ----------

=======
# MAGIC Table names come from Databricks widgets. They are used in SQL
# MAGIC identifiers rather than SQL parameters, so validate them before using
# MAGIC them in queries.

# COMMAND ----------

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, name: str) -> str:
    """Validate a PostgreSQL table identifier supplied through a widget."""
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Invalid {name}: {value!r}. "
            "Only letters, numbers, and underscores are allowed, "
            "and the name must not begin with a number."
        )
    return value


WEATHER_TABLE_NAME = validate_identifier(
    WEATHER_TABLE_NAME,
    "weather_table_name",
)

EMBEDDINGS_TABLE_NAME = validate_identifier(
    EMBEDDINGS_TABLE_NAME,
    "embeddings_table_name",
)

print("Table names validated successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## pg8000 Connection Helper
# MAGIC
# MAGIC `pg8000.native.Connection` is used instead of psycopg2.
# MAGIC
# MAGIC SSL is enabled using:
# MAGIC
# MAGIC ```python
# MAGIC ssl_context=True
# MAGIC ```
# MAGIC
# MAGIC which provides the equivalent secure connection behavior needed for
# MAGIC the Lakebase PostgreSQL endpoint.

# COMMAND ----------

# DBTITLE 1,Define Lakebase Connection Functions
>>>>>>> Stashed changes
def get_connection():
    """Return a new pg8000 native Lakebase connection.
    
    Uses native Postgres password authentication from the secret.
    OAuth authentication is not configured for this endpoint.
    """
<<<<<<< Updated upstream
    Create a pg8000 connection using a freshly generated Lakebase credential.

    IMPORTANT:
    This function intentionally does not use:
        - psycopg2
        - lakebase.py
        - a stored database password
        - the legacy Database Instance credential API
    """

    credential = generate_lakebase_credential()

    # The credential returned by the Databricks PostgreSQL API provides
    # the database username and password/token needed by the endpoint.
    #
    # Different SDK versions expose these fields slightly differently,
    # so handle the common representations explicitly.

    if hasattr(credential, "token"):
        password = credential.token
    elif hasattr(credential, "password"):
        password = credential.password
    elif isinstance(credential, dict):
        password = (
            credential.get("token")
            or credential.get("password")
        )
    else:
        raise RuntimeError(
            "Could not determine the Lakebase credential token "
            "from the Databricks SDK response."
        )

    if not password:
        raise RuntimeError(
            "Lakebase credential was generated but no token/password "
            "was returned."
        )

    # Lakebase PostgreSQL endpoint.
    #
    # The endpoint hostname used by the project is resolved from the
    # endpoint resource. For the current Weather Intelligence project,
    # this is the standard Lakebase endpoint hostname.
    #
    # If your endpoint uses a different hostname, set these values in
    # the configuration cell rather than changing the authentication
    # mechanism.

    db_host = os.environ.get(
        "LAKEBASE_DB_HOST"
    )

    db_port = int(
        os.environ.get(
            "LAKEBASE_DB_PORT",
            "5432"
        )
    )

    db_name = os.environ.get(
        "LAKEBASE_DB_NAME",
        "databricks_postgres"
    )

    db_user = os.environ.get(
        "LAKEBASE_DB_USER",
        "student"
    )

    if not db_host:
        raise RuntimeError(
            "LAKEBASE_DB_HOST is not configured.\n\n"
            "Set the PostgreSQL hostname for your Lakebase endpoint "
            "before running this notebook."
        )

    return pg8000.native.Connection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=password,
        ssl_context=True,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connection Configuration
# MAGIC
# MAGIC IMPORTANT:
# MAGIC
# MAGIC The Lakebase PostgreSQL endpoint resource name and the PostgreSQL
# MAGIC hostname are two different values.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```text
# MAGIC Endpoint resource:
# MAGIC projects/dataexpert-student/branches/production/endpoints/primary
# MAGIC
# MAGIC PostgreSQL host:
# MAGIC ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com
# MAGIC ```
# MAGIC
# MAGIC The endpoint resource is used to generate the credential.
# MAGIC
# MAGIC The hostname is used by pg8000 to establish the PostgreSQL connection.

# COMMAND ----------

# If these environment variables are not already configured,
# set them here for the notebook.
#
# IMPORTANT:
# Replace the values below only if your Lakebase endpoint differs.

DB_HOST = os.environ.get(
    "LAKEBASE_DB_HOST",
    "ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com"
)

DB_PORT = int(
    os.environ.get(
        "LAKEBASE_DB_PORT",
        "5432"
    )
)

DB_NAME = os.environ.get(
    "LAKEBASE_DB_NAME",
    "databricks_postgres"
)

DB_USER = os.environ.get(
    "LAKEBASE_DB_USER",
    "student"
)

os.environ["LAKEBASE_DB_HOST"] = DB_HOST
os.environ["LAKEBASE_DB_PORT"] = str(DB_PORT)
os.environ["LAKEBASE_DB_NAME"] = DB_NAME
os.environ["LAKEBASE_DB_USER"] = DB_USER

print("Lakebase PostgreSQL configuration")
print("=" * 70)
print(f"Host:     {DB_HOST}")
print(f"Port:     {DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User:     {DB_USER}")
print(f"Endpoint: {LAKEBASE_ENDPOINT_NAME}")
=======
    return pg8000.native.Connection(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        ssl_context=True,
    )


def query_rows(sql: str, **params):
    """
    Execute a SELECT statement with pg8000 and return dictionaries.

    pg8000.native returns rows as tuples. We retrieve the column names from
    the result metadata and convert each row into a dictionary so the rest
    of the notebook can use named fields.
    """
    conn = get_connection()

    try:
        result = conn.run(sql, **params)

        if not result:
            return []

        columns = [column["name"] for column in conn.columns]

        return [
            dict(zip(columns, row))
            for row in result
        ]

    finally:
        conn.close()


def execute_sql(sql: str, **params):
    """Execute a write/DDL statement using pg8000."""
    conn = get_connection()

    try:
        result = conn.run(sql, **params)
        conn.commit()
        return result

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Lakebase Connection
<<<<<<< Updated upstream
# MAGIC
# MAGIC This test generates a **fresh credential** immediately before connecting.
# MAGIC
# MAGIC If this fails with:
# MAGIC
# MAGIC ```text
# MAGIC External authorization failed
# MAGIC ```
# MAGIC
# MAGIC the problem is no longer stale credentials stored in a secret.
# MAGIC
# MAGIC The likely causes then become:
# MAGIC
# MAGIC - incorrect Lakebase endpoint resource
# MAGIC - endpoint paused/unavailable
# MAGIC - Databricks App/compute identity does not have access
# MAGIC - network/IP ACL restrictions
# MAGIC - incorrect PostgreSQL hostname
# MAGIC - incorrect database user

# COMMAND ----------

=======

# COMMAND ----------

# DBTITLE 1,Test Lakebase Connection
>>>>>>> Stashed changes
try:
    conn = get_connection()
    
    try:
        result = conn.run(
            """
            SELECT
                current_database(),
                current_user
            """
        )

        database, user = result[0]

        print("Lakebase connection successful.")
        print(f"Database: {database}")
        print(f"User:     {user}")
        print(f"Host:     {DB_HOST}:{DB_PORT}")

    finally:
        conn.close()
        
except Exception as e:
<<<<<<< Updated upstream

    print("Lakebase connection FAILED.")
    print()
    print(f"Error: {e}")
    print()
    print("Connection details:")
    print(f"  Endpoint: {LAKEBASE_ENDPOINT_NAME}")
    print(f"  Host:     {DB_HOST}")
    print(f"  Port:     {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User:     {DB_USER}")
    print()
    print(
        "The notebook generated a fresh credential using "
        "WorkspaceClient.postgres.generate_database_credential()."
    )

=======
    print(f"❌ Connection failed: {e}")
    print(f"\nConnection details:")
    print(f"  Host: {DB_HOST}")
    print(f"  Port: {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User: {DB_USER}")
    
    error_str = str(e).lower()
    
    if "external authorization failed" in error_str:
        print(f"\n⚠️  External authorization failed.")
        print(f"\nThis typically means:")
        print(f"  1. The credentials in the secret are stale/expired (OAuth tokens expire after 1 hour)")
        print(f"  2. The user/role doesn't have CONNECT permission")
        print(f"  3. IP ACL restrictions are blocking this compute")
        print(f"\nTo fix:")
        print(f"  1. Generate fresh credentials using the Databricks SDK:")
        print(f"     ```python")
        print(f"     from databricks.sdk import WorkspaceClient")
        print(f"     w = WorkspaceClient()")
        print(f"     creds = w.postgres.generate_database_credential(")
        print(f"         endpoint='projects/dataexpert-student/branches/production/endpoints/primary'")
        print(f"     )")
        print(f"     # Then update the secret with the new credentials")
        print(f"     ```")
        print(f"  2. Or switch to using SDK-generated credentials directly instead of the secret")
    
>>>>>>> Stashed changes
    raise

# COMMAND ----------

# DBTITLE 1,Diagnosis: OAuth Not Configured
# MAGIC %md
# MAGIC ## ⚠️ Diagnosis: OAuth Authentication Not Configured
# MAGIC
# MAGIC **Root cause:** The Lakebase role `mpdataanlayst` has no spec/configuration. OAuth authentication cannot work without a properly configured role mapping.
# MAGIC
# MAGIC **What happened:**
# MAGIC - The original secret used **native Postgres password authentication** (username: `student`, password: `<password>`)
# MAGIC - Cell 19 overwrote the secret with OAuth credentials
# MAGIC - OAuth credentials don't work because the role isn't configured
# MAGIC - Now the secret contains non-working OAuth credentials
# MAGIC
# MAGIC **To fix this:**
# MAGIC
# MAGIC 1. **Option A: Restore original native password credentials** (RECOMMENDED)
# MAGIC    - Ask a workspace admin to restore the original secret with native password auth
# MAGIC    - Or find the original password and manually update the secret
# MAGIC
# MAGIC 2. **Option B: Configure OAuth properly**
# MAGIC    - Have an admin configure the Lakebase role with proper identity mapping
# MAGIC    - Then OAuth tokens will work
# MAGIC
# MAGIC 3. **Option C: Test on a different compute**
# MAGIC    - Try connecting from a regular interactive cluster (not Serverless GPU)
# MAGIC    - Rules out network access issues specific to Serverless GPU
# MAGIC
# MAGIC The Serverless GPU compute may also have network restrictions preventing access to this Lakebase endpoint.

# COMMAND ----------

# DBTITLE 1,Solution: Restore Original Secret
# If you have the original password, run this to restore the secret
# Otherwise, ask a workspace admin to restore it

from databricks.sdk import WorkspaceClient
import base64
from urllib.parse import quote_plus

w = WorkspaceClient()

# REPLACE THIS with the original password for the 'student' user
ORIGINAL_PASSWORD = "<REPLACE_WITH_ORIGINAL_PASSWORD>"

if ORIGINAL_PASSWORD == "<REPLACE_WITH_ORIGINAL_PASSWORD>":
    print("⚠️  You need to provide the original password.")
    print("\nIf you don't have it:")
    print("  1. Ask the workspace admin who set up this Lakebase instance")
    print("  2. Or check if there's a backup of the secret")
    print("  3. Or the admin can reset the password in Lakebase console")
else:
    # Construct the original PostgreSQL URL with native password auth
    original_url = (
        f"postgresql://student:{quote_plus(ORIGINAL_PASSWORD)}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    )
    
    # Base64-encode for the secret
    encoded_url = base64.b64encode(original_url.encode('utf-8')).decode('utf-8')
    
    try:
        w.secrets.put_secret(
            scope=DATABASE_SECRET_SCOPE,
            key=DATABASE_SECRET_KEY,
            string_value=encoded_url
        )
        
        print("✅ Secret restored with native password authentication!")
        print("\nNext steps:")
        print("  1. Re-run cell 10 to reload the credentials from the secret")
        print("  2. Re-run cell 16 to test the connection")
        
    except Exception as e:
        print(f"❌ Failed to update secret: {e}")

# COMMAND ----------

# Test pg8000 connection
# NOTE: Using pg8000 instead of psycopg2 for Serverless compatibility
# - psycopg2-binary has native C extensions that crash on Serverless (SIGABRT 134)
# - pg8000 is pure Python with no C dependencies, works reliably on Serverless
import pg8000.native

try:
    # Connect using pg8000 (pure Python, works on Serverless)
    conn = pg8000.native.Connection(
        host=db_host,
        port=parsed.port or 5432,
        database=db_name,
        user=parsed.username,
        password=parsed.password,
        ssl_context=True  # Equivalent to sslmode=require
    )
    
    # Test query to count rows in watchlist table
    result = conn.run(f"SELECT COUNT(*) as count FROM {WATCHLIST_TABLE_NAME}")
    count = result[0][0] if result else 0
    
    print(f"✅ pg8000 connection successful! Found {count} rows in {WATCHLIST_TABLE_NAME}")
    
    # Show sample rows
    if count > 0:
        rows = conn.run(f"SELECT * FROM {WATCHLIST_TABLE_NAME} LIMIT 5")
        print(f"\nSample rows:")
        for row in rows:
            print(f"  {row}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ pg8000 connection failed: {e}")

# COMMAND ----------

# Test the Lakebase connection
try:
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            database, user = cur.fetchone()
    
    print("✓ Lakebase connection successful")
    print(f"  Database: {database}")
    print(f"  User:     {user}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nPossible causes:")
    print("  • Lakebase instance is paused (resume it in the Lakebase console)")
    print("  • IP ACL restrictions blocking this compute")
    print("  • Network/private link configuration issue")
    print("  • Invalid credentials in the secret")
    raise


# COMMAND ----------

# DBTITLE 1,Diagnose Lakebase Configuration
# Diagnostic: Check Lakebase configuration
import base64
import os
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get the secret
scope = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
key = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

try:
    secret = w.secrets.get_secret(scope=scope, key=key)
    lakebase_url = base64.b64decode(secret.value).decode("utf-8")
    parsed = urlparse(lakebase_url)
    
    print("✓ Secret found and decoded")
    print(f"  Scope: {scope}")
    print(f"  Key: {key}")
    print(f"\nConnection details:")
    print(f"  Host: {parsed.hostname}")
    print(f"  Port: {parsed.port or 5432}")
    print(f"  Database: {parsed.path.lstrip('/')}")
    print(f"  Username: {parsed.username}")
    
    # Extract project and branch from hostname
    # Format: ep-{branch-name}-{random}.{project-name}.{region}.cloud.databricks.com
    hostname_parts = parsed.hostname.split('.')
    if len(hostname_parts) >= 2:
        project_name = hostname_parts[1]
        branch_part = hostname_parts[0]
        print(f"\nInferred from hostname:")
        print(f"  Project: {project_name}")
        print(f"  Endpoint: {branch_part}")
    
    print(f"\n⚠️ Connection Error: External authorization failed")
    print(f"\nThis means the Lakebase instance exists but is not accessible.")
    print(f"\nTo fix this, you need to:")
    print(f"  1. Check if the Lakebase project '{project_name}' exists")
    print(f"  2. Check if the branch/endpoint is READY (not PAUSED or ARCHIVED)")
    print(f"  3. Verify this compute has network access to the endpoint")
    print(f"  4. Check IP ACL settings if configured")
    print(f"\nRun this command to check the project status:")
    print(f"  from databricks.sdk import WorkspaceClient")
    print(f"  w = WorkspaceClient()")
    print(f"  project = w.postgres.get_project(name='projects/{project_name}')")
    print(f"  print(project.status.current_state)")
    
except Exception as e:
    print(f"✗ Error: {e}")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
<<<<<<< Updated upstream
# MAGIC The Lakebase database should contain:
# MAGIC
# MAGIC ```text
# MAGIC weather_news
# MAGIC weather_embeddings
# MAGIC ```
# MAGIC
# MAGIC Expected `weather_news` columns:
=======
# MAGIC The SQL setup files should already have created:
# MAGIC
# MAGIC ### `weather_news`
# MAGIC
# MAGIC Expected columns:
>>>>>>> Stashed changes
# MAGIC
# MAGIC - id
# MAGIC - location
# MAGIC - source_type
# MAGIC - headline
# MAGIC - narrative_text
# MAGIC - issued_at
# MAGIC - effective_at
# MAGIC - payload
# MAGIC - synced_at
# MAGIC
# MAGIC ### `weather_embeddings`
# MAGIC
# MAGIC Expected columns:
# MAGIC
# MAGIC - id
# MAGIC - document_id
# MAGIC - chunk_index
# MAGIC - chunk_text
# MAGIC - embedding
# MAGIC - model_name
# MAGIC - created_at

# COMMAND ----------

<<<<<<< Updated upstream
conn = get_connection()

try:

    result = conn.run(
        """
        SELECT
            table_name,
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns
        WHERE table_name IN (:weather_table, :embedding_table)
        ORDER BY table_name, ordinal_position
        """,
        weather_table=WEATHER_TABLE_NAME,
        embedding_table=EMBEDDINGS_TABLE_NAME,
    )

finally:

    conn.close()

print("Database schema")
print("=" * 70)

for row in result:

    table_name = row[0]
    column_name = row[1]
    data_type = row[2]
    udt_name = row[3]

=======
table_columns = query_rows(
    """
    SELECT
        table_name,
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_name IN (:weather_table, :embeddings_table)
    ORDER BY table_name, ordinal_position
    """,
    weather_table=WEATHER_TABLE_NAME,
    embeddings_table=EMBEDDINGS_TABLE_NAME,
)

if not table_columns:
    raise RuntimeError(
        f"No columns found for {WEATHER_TABLE_NAME!r} or "
        f"{EMBEDDINGS_TABLE_NAME!r}. "
        "Run the required SQL setup files first."
    )

for row in table_columns:
>>>>>>> Stashed changes
    print(
        f"{row['table_name']}.{row['column_name']}: "
        f"{row['data_type']} ({row['udt_name']})"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embedding Column
# MAGIC
# MAGIC The embedding column must be a PostgreSQL `vector` column.
# MAGIC
# MAGIC The expected configuration is:
# MAGIC
# MAGIC ```text
# MAGIC vector(384)
# MAGIC ```
# MAGIC
# MAGIC The SQL setup should also create an HNSW index using
# MAGIC `vector_cosine_ops`.

# COMMAND ----------

embedding_column = query_rows(
    """
    SELECT
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_name = :table_name
      AND column_name = 'embedding'
    """,
    table_name=EMBEDDINGS_TABLE_NAME,
)

if not embedding_column:
    raise RuntimeError(
        f"The {EMBEDDINGS_TABLE_NAME}.embedding column was not found."
    )

print(f"Embedding data type: {embedding_column[0]['data_type']}")
print(f"Embedding UDT:       {embedding_column[0]['udt_name']}")

if embedding_column[0]["udt_name"] != "vector":
    raise RuntimeError(
        f"{EMBEDDINGS_TABLE_NAME}.embedding is not a pgvector column. "
        f"Expected udt_name='vector', received "
        f"{embedding_column[0]['udt_name']!r}."
    )

print("✅ pgvector column verified.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Weather Documents
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint should be run before
# MAGIC this notebook.
# MAGIC
# MAGIC Example request:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "locations": ["Chicago, IL", "Austin, TX"],
# MAGIC   "limit": 50
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC The endpoint stores normalized NWS alerts and forecasts in
# MAGIC `weather_news`.

# COMMAND ----------

<<<<<<< Updated upstream
conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT COUNT(*)
        FROM {WEATHER_TABLE_NAME}
        """
    )

    total_documents = result[0][0]

    result = conn.run(
        f"""
        SELECT COUNT(*)
        FROM {WEATHER_TABLE_NAME}
        WHERE narrative_text IS NOT NULL
          AND TRIM(narrative_text) <> ''
        """
    )

    documents_with_text = result[0][0]

finally:

    conn.close()

print(
    f"Total weather documents:   {total_documents}"
)
=======
document_counts = query_rows(
    f"""
    SELECT
        COUNT(*) AS total_documents,
        COUNT(*) FILTER (
            WHERE narrative_text IS NOT NULL
              AND TRIM(narrative_text) <> ''
        ) AS documents_with_text
    FROM {WEATHER_TABLE_NAME}
    """
)

total_documents = document_counts[0]["total_documents"]
documents_with_text = document_counts[0]["documents_with_text"]
>>>>>>> Stashed changes

print(
    f"Documents containing text: {documents_with_text}"
)

if total_documents == 0:
    print(
        "\n⚠️ No weather documents are currently available.\n"
        "Run POST /weather/sync from the Flask application first."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Unembedded Weather Documents
# MAGIC
# MAGIC A document is considered unembedded when no corresponding
# MAGIC `weather_embeddings` row exists.
# MAGIC
<<<<<<< Updated upstream
# MAGIC This makes the notebook safe to run repeatedly.

# COMMAND ----------

conn = get_connection()

try:

    documents = conn.run(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.narrative_text,
            d.issued_at,
            d.effective_at
        FROM {WEATHER_TABLE_NAME} d
        WHERE d.narrative_text IS NOT NULL
          AND TRIM(d.narrative_text) <> ''
          AND NOT EXISTS (
              SELECT 1
              FROM {EMBEDDINGS_TABLE_NAME} e
              WHERE e.document_id = d.id
          )
        ORDER BY d.synced_at ASC
        """
    )

finally:

    conn.close()

# Convert pg8000 tuples into dictionaries.

document_columns = [
    "id",
    "location",
    "source_type",
    "headline",
    "narrative_text",
    "issued_at",
    "effective_at",
]

documents = [
    dict(zip(document_columns, row))
    for row in documents
]

print(
    f"Found {len(documents)} unembedded weather documents."
)

=======
# MAGIC This makes the notebook safe to run repeatedly:
# MAGIC
# MAGIC - previously embedded documents are skipped
# MAGIC - newly synchronized NWS documents are picked up
# MAGIC - existing embeddings are not unnecessarily duplicated
# MAGIC
# MAGIC Because one document may produce multiple chunks, the existence check
# MAGIC is performed at the document level.

# COMMAND ----------

documents = query_rows(
    f"""
    SELECT
        d.id,
        d.location,
        d.source_type,
        d.headline,
        d.narrative_text,
        d.issued_at,
        d.effective_at,
        d.synced_at
    FROM {WEATHER_TABLE_NAME} d
    WHERE d.narrative_text IS NOT NULL
      AND TRIM(d.narrative_text) <> ''
      AND NOT EXISTS (
          SELECT 1
          FROM {EMBEDDINGS_TABLE_NAME} e
          WHERE e.document_id = d.id
      )
    ORDER BY d.synced_at ASC
    """,
)

print(f"Found {len(documents)} unembedded weather documents.")

>>>>>>> Stashed changes
if documents:
    print("\nSample documents:")

    for document in documents[:5]:
        narrative = document["narrative_text"] or ""

        print()
        print(f"ID:       {document['id']}")
        print(f"Location: {document['location']}")
        print(f"Type:     {document['source_type']}")
        print(f"Headline: {document['headline']}")
        print(
<<<<<<< Updated upstream
            f"Text:     "
            f"{document['narrative_text'][:300]}..."
=======
            f"\nID:       {document['id']}"
            f"\nLocation: {document['location']}"
            f"\nType:     {document['source_type']}"
            f"\nHeadline: {document['headline']}"
            f"\nText:     {narrative[:300]}..."
>>>>>>> Stashed changes
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
<<<<<<< Updated upstream
# MAGIC NWS narrative text is split into overlapping character-based chunks.
# MAGIC
# MAGIC Default:
# MAGIC
# MAGIC ```text
# MAGIC Chunk size:    800
# MAGIC Chunk overlap: 100
# MAGIC ```
=======
# MAGIC NWS narrative text is generally short, but alerts can contain longer
# MAGIC descriptions and instructions.
# MAGIC
# MAGIC We use the assignment-recommended:
# MAGIC
# MAGIC ```text
# MAGIC CHUNK_SIZE    = 800 characters
# MAGIC CHUNK_OVERLAP = 100 characters
# MAGIC ```
# MAGIC
# MAGIC Most NWS documents will remain a single chunk. Longer alert narratives
# MAGIC are split into overlapping windows.
>>>>>>> Stashed changes

# COMMAND ----------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if not text:

        return []

    text = text.strip()

    if not text:

        return []

    if len(text) <= chunk_size:

        return [text]

    step = chunk_size - overlap
    chunks = []

    for start in range(
        0,
        len(text),
        step,
    ):

        chunk = text[start:start + chunk_size].strip()

        if chunk:

            chunks.append(chunk)

        if start + chunk_size >= len(text):

            break

    return chunks

<<<<<<< Updated upstream

# COMMAND ----------
=======
>>>>>>> Stashed changes

chunk_rows = []

for document in documents:

    chunks = chunk_text(
        document["narrative_text"],
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    for chunk_index, text in enumerate(chunks):

        chunk_rows.append(
            {
                "document_id": document["id"],
                "location": document["location"],
                "headline": document["headline"],
                "source_type": document["source_type"],
                "chunk_index": chunk_index,
                "chunk_text": text,
            }
        )

<<<<<<< Updated upstream
print(
    f"Created {len(chunk_rows)} text chunks."
)

if chunk_rows:

    print()
    print("Sample chunk:")
    print(chunk_rows[0])
=======
print(f"Created {len(chunk_rows)} text chunks.")

if chunk_rows:

    print("\nSample chunk:")
    print(f"Document: {chunk_rows[0]['document_id']}")
    print(f"Chunk:    {chunk_rows[0]['chunk_index']}")
    print(f"Text:     {chunk_rows[0]['chunk_text'][:500]}")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Model
# MAGIC
<<<<<<< Updated upstream
# MAGIC `all-MiniLM-L6-v2` produces 384-dimensional vectors.
=======
# MAGIC The same model is used by the Flask `/weather/search` endpoint.
# MAGIC
# MAGIC `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional
# MAGIC embeddings.
>>>>>>> Stashed changes

# COMMAND ----------

from sentence_transformers import SentenceTransformer

print(f"Loading {EMBEDDING_MODEL_NAME}...")

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("✅ Embedding model loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Chunk Embeddings
# MAGIC
# MAGIC Embeddings are generated in batches to control memory usage.
# MAGIC
<<<<<<< Updated upstream
# MAGIC Embeddings are normalized so cosine similarity can be used consistently
# MAGIC during pgvector search.
=======
# MAGIC `normalize_embeddings=True` normalizes the vectors. This is compatible
# MAGIC with cosine similarity using pgvector's `<=>` operator.
>>>>>>> Stashed changes

# COMMAND ----------

embedded_rows = []

if not chunk_rows:

    print("No chunks require embedding.")

else:

    for start in range(0, len(chunk_rows), BATCH_SIZE):

        batch = chunk_rows[start:start + BATCH_SIZE]

        texts = [
            row["chunk_text"]
            for row in batch
        ]

<<<<<<< Updated upstream
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional "
                f"vector but received "
                f"{len(vector)} dimensions."
            )

        row_with_embedding = dict(row)

        row_with_embedding[
            "embedding"
        ] = vector.tolist()
=======
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
>>>>>>> Stashed changes

        for row, vector in zip(batch, vectors):

            vector_dimension = len(vector)

            if vector_dimension != EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {EMBEDDING_DIM}-dimensional vector but "
                    f"received {vector_dimension} dimensions."
                )

            row_with_embedding = dict(row)

            row_with_embedding["embedding"] = [
                float(value)
                for value in vector
            ]

            embedded_rows.append(row_with_embedding)

        processed = min(
            start + BATCH_SIZE,
            len(chunk_rows),
        )

<<<<<<< Updated upstream
    print(
        f"Embedded "
        f"{min(start + BATCH_SIZE, len(chunk_rows))}"
        f"/{len(chunk_rows)} chunks"
    )

print()
print(
    f"Generated {len(embedded_rows)} embeddings."
)
=======
        print(
            f"Embedded {processed}/{len(chunk_rows)} chunks"
        )

print(f"\nGenerated {len(embedded_rows)} embeddings.")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embedding
# MAGIC
# MAGIC This confirms that the embedding has the expected 384 dimensions before
# MAGIC anything is written to Lakebase.

# COMMAND ----------

if embedded_rows:

    sample = embedded_rows[0]

<<<<<<< Updated upstream
    print(
        f"Document ID:       "
        f"{sample['document_id']}"
    )

    print(
        f"Chunk index:       "
        f"{sample['chunk_index']}"
    )

    print(
        f"Chunk text:        "
        f"{sample['chunk_text'][:500]}"
    )

    print(
        f"Vector dimensions: "
        f"{len(sample['embedding'])}"
    )

    print(
        f"Model:             "
        f"{EMBEDDING_MODEL_NAME}"
    )
=======
    print(f"Document ID:       {sample['document_id']}")
    print(f"Chunk index:       {sample['chunk_index']}")
    print(f"Chunk text:        {sample['chunk_text'][:500]}")
    print(f"Vector dimensions: {len(sample['embedding'])}")
    print(f"Model:             {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Vector Values
# MAGIC
# MAGIC `pg8000` does not require Spark JDBC or a Spark-specific
# MAGIC `stringtype=unspecified` workaround.
# MAGIC
# MAGIC Each embedding is converted to a PostgreSQL vector literal:
# MAGIC
# MAGIC ```text
# MAGIC [0.0123,-0.0345,...]
# MAGIC ```
# MAGIC
# MAGIC and explicitly cast in SQL:
# MAGIC
# MAGIC ```sql
# MAGIC :embedding::vector
# MAGIC ```
# MAGIC
# MAGIC This writes directly into the `VECTOR(384)` column.

# COMMAND ----------

insert_rows = []

for row in embedded_rows:

    embedding_id = (
        f"{row['document_id']}_{row['chunk_index']}"
    )

    vector_string = (
        "["
        + ",".join(
            str(float(value))
            for value in row["embedding"]
        )
        + "]"
    )

    insert_rows.append(
        {
            "id": embedding_id,
            "document_id": row["document_id"],
            "chunk_index": int(row["chunk_index"]),
            "chunk_text": row["chunk_text"],
            "embedding": vector_string,
            "model_name": EMBEDDING_MODEL_NAME,
        }
    )

print(f"Prepared {len(insert_rows)} rows for Lakebase.")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
<<<<<<< Updated upstream
# MAGIC Embeddings are written directly using pg8000.
# MAGIC
# MAGIC No Spark JDBC write is used.
# MAGIC
# MAGIC No psycopg2 is used.
=======
# MAGIC Embeddings are written directly through `pg8000`.
# MAGIC
# MAGIC No Spark JDBC is used.
# MAGIC
# MAGIC No `psycopg2` is used.
# MAGIC
# MAGIC No post-processing array-to-vector conversion is required.
>>>>>>> Stashed changes
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC ```text
# MAGIC document_id + "_" + chunk_index
<<<<<<< Updated upstream
# MAGIC ```
# MAGIC
# MAGIC The vector is explicitly cast:
# MAGIC
# MAGIC ```sql
# MAGIC CAST(:embedding AS vector)
# MAGIC ```
# MAGIC
# MAGIC This preserves the PostgreSQL `vector` type.

# COMMAND ----------

if not embedded_rows:

    print(
        "No new embeddings to write."
    )
=======
# MAGIC ```
# MAGIC
# MAGIC The `weather_embeddings.id` primary key makes the operation safely
# MAGIC repeatable.
# MAGIC
# MAGIC The `ON CONFLICT` clause updates the existing chunk if the source text
# MAGIC or embedding changes.

# COMMAND ----------

if not insert_rows:

    print("No new embeddings to write.")
>>>>>>> Stashed changes

else:

    insert_sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE_NAME} (
            id,
            document_id,
            chunk_index,
            chunk_text,
            embedding,
            model_name,
            created_at
        )
        VALUES (
            :id,
            :document_id,
            :chunk_index,
            :chunk_text,
            :embedding::vector,
            :model_name,
            now()
        )
        ON CONFLICT (id) DO UPDATE
        SET
            document_id = EXCLUDED.document_id,
            chunk_index = EXCLUDED.chunk_index,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = CURRENT_TIMESTAMP
    """

    total_written = 0

    conn = get_connection()

    try:

        for start in range(
            0,
            len(insert_rows),
            BATCH_SIZE,
        ):

            batch = insert_rows[
                start:start + BATCH_SIZE
            ]

            for row in batch:

                conn.run(
                    insert_sql,
<<<<<<< Updated upstream
                    id=embedding_id,
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    chunk_text=row["chunk_text"],
                    embedding=vector_string,
                    model_name=EMBEDDING_MODEL_NAME,
=======
                    **row,
>>>>>>> Stashed changes
                )

                total_written += 1

            conn.commit()

            processed = min(
                start + BATCH_SIZE,
                len(insert_rows),
            )

<<<<<<< Updated upstream
=======
            print(
                f"Wrote {processed}/{len(insert_rows)} embeddings"
            )

    except Exception:

        conn.rollback()
        raise

>>>>>>> Stashed changes
    finally:

        conn.close()

<<<<<<< Updated upstream
    print()
    print(
        f"Successfully wrote "
        f"{total_written} embeddings "
=======
    print(
        f"\n✅ Successfully wrote {total_written} embeddings "
>>>>>>> Stashed changes
        f"to {EMBEDDINGS_TABLE_NAME}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings
# MAGIC
# MAGIC Confirm:
# MAGIC
# MAGIC 1. Embeddings exist.
<<<<<<< Updated upstream
# MAGIC 2. The expected number of documents have embeddings.
# MAGIC 3. The `embedding` column is PostgreSQL `vector`.
# MAGIC 4. The vector dimension is 384.

# COMMAND ----------

conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT document_id)
        FROM {EMBEDDINGS_TABLE_NAME}
        """
    )

    total_embeddings = result[0][0]
    documents_embedded = result[0][1]

    result = conn.run(
        """
        SELECT
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns
        WHERE table_name = :table_name
          AND column_name = 'embedding'
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

    embedding_column = result[0] if result else None

finally:

    conn.close()

print(
    f"Total embeddings:   {total_embeddings}"
=======
# MAGIC 2. Documents have corresponding chunks.
# MAGIC 3. The embedding column is actually a pgvector `vector` type.

# COMMAND ----------

verification = query_rows(
    f"""
    SELECT
        COUNT(*) AS total_embeddings,
        COUNT(DISTINCT document_id) AS documents_embedded
    FROM {EMBEDDINGS_TABLE_NAME}
    """
>>>>>>> Stashed changes
)

print(
    f"Total embeddings:   {verification[0]['total_embeddings']}"
)

<<<<<<< Updated upstream
if embedding_column:

    print(
        f"Embedding column:   "
        f"{embedding_column[0]}"
    )

    print(
        f"Data type:          "
        f"{embedding_column[1]}"
    )

    print(
        f"UDT:                "
        f"{embedding_column[2]}"
    )
=======
print(
    f"Documents embedded: {verification[0]['documents_embedded']}"
)

# COMMAND ----------

embedding_column = query_rows(
    """
    SELECT
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_name = :table_name
      AND column_name = 'embedding'
    """,
    table_name=EMBEDDINGS_TABLE_NAME,
)

if embedding_column:

    print(
        f"Embedding data type: "
        f"{embedding_column[0]['data_type']}"
    )

    print(
        f"Embedding UDT:       "
        f"{embedding_column[0]['udt_name']}"
    )

    if embedding_column[0]["udt_name"] == "vector":
        print("✅ Embedding column is a pgvector VECTOR column.")
    else:
        print("⚠️ Embedding column is not reported as VECTOR.")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
<<<<<<< Updated upstream
# MAGIC ## Verify Vector Dimension
# MAGIC
# MAGIC PostgreSQL/pgvector should report a 384-dimensional vector.

# COMMAND ----------

conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT
            vector_dims(embedding)
        FROM {EMBEDDINGS_TABLE_NAME}
        WHERE embedding IS NOT NULL
        LIMIT 1
        """
    )

finally:

    conn.close()

if result:

    vector_dimension = result[0][0]

    print(
        f"Vector dimension: {vector_dimension}"
    )

    if vector_dimension != EMBEDDING_DIM:

        raise ValueError(
            f"Expected vector dimension "
            f"{EMBEDDING_DIM}, but database "
            f"contains {vector_dimension}."
        )

    print(
        "Vector dimension validation passed."
    )
=======
# MAGIC ## Verify HNSW Index
# MAGIC
# MAGIC The SQL setup should have created an HNSW index using
# MAGIC `vector_cosine_ops`.
# MAGIC
# MAGIC This is the index used by pgvector to accelerate cosine-distance
# MAGIC retrieval.

# COMMAND ----------

indexes = query_rows(
    f"""
    SELECT
        indexname,
        indexdef
    FROM pg_indexes
    WHERE tablename = :table_name
    ORDER BY indexname
    """,
    table_name=EMBEDDINGS_TABLE_NAME,
)

if indexes:

    for index in indexes:

        print(f"\nIndex: {index['indexname']}")
        print(index["indexdef"])
>>>>>>> Stashed changes

else:

    print(
<<<<<<< Updated upstream
        "No vectors available for dimension validation."
=======
        f"No indexes found for {EMBEDDINGS_TABLE_NAME}. "
        "Verify that the embedding SQL setup was executed."
>>>>>>> Stashed changes
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
<<<<<<< Updated upstream
# MAGIC This validates that pgvector can perform cosine-distance search.
=======
# MAGIC This validation query confirms that:
# MAGIC
# MAGIC - the embedding column is queryable as a vector
# MAGIC - the `<=>` cosine-distance operator works
# MAGIC - the vector can be ranked by similarity
>>>>>>> Stashed changes
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC ```text
# MAGIC POST /weather/search
# MAGIC ```
# MAGIC
<<<<<<< Updated upstream
# MAGIC The query vector must be generated with the same
# MAGIC `all-MiniLM-L6-v2` model.
=======
# MAGIC The Flask endpoint embeds a user's query using the same model and
# MAGIC searches this table using cosine distance.
>>>>>>> Stashed changes

# COMMAND ----------

if embedded_rows:

    test_vector = embedded_rows[0]["embedding"]

    test_vector_string = (
        "["
        + ",".join(
            str(float(value))
            for value in test_vector
        )
        + "]"
    )

<<<<<<< Updated upstream
    conn = get_connection()

    try:

        results = conn.run(
            f"""
            SELECT
                e.id,
                e.document_id,
                e.chunk_index,
                e.chunk_text,
                1 - (
                    e.embedding
                    <=>
                    CAST(:query_vector AS vector)
                ) AS similarity
            FROM {EMBEDDINGS_TABLE_NAME} e
            ORDER BY
                e.embedding
                <=>
                CAST(:query_vector AS vector)
            LIMIT 5
            """,
            query_vector=test_vector_string,
        )

    finally:

        conn.close()

    print(
        "Top 5 cosine similarity results:"
    )

    print()

    for result in results:

        embedding_id = result[0]
        document_id = result[1]
        chunk_index = result[2]
        chunk_text = result[3]
        similarity = result[4]

=======
    results = query_rows(
        f"""
        SELECT
            e.id,
            e.document_id,
            e.chunk_index,
            e.chunk_text,
            1 - (
                e.embedding <=> :query_vector::vector
            ) AS similarity
        FROM {EMBEDDINGS_TABLE_NAME} e
        ORDER BY
            e.embedding <=> :query_vector::vector
        LIMIT 5
        """,
        query_vector=test_vector_string,
    )

    print("Top 5 cosine-similarity results:\n")

    for result in results:

>>>>>>> Stashed changes
        print(
            f"Similarity: {float(result['similarity']):.4f}"
        )

        print(
<<<<<<< Updated upstream
            f"Embedding:  {embedding_id}"
        )

        print(
            f"Document:   {document_id}"
=======
            f"Document:   {result['document_id']}"
>>>>>>> Stashed changes
        )

        print(
            f"Chunk:      {result['chunk_index']}"
        )

        print(
            f"Text:       {result['chunk_text'][:300]}"
        )

        print("-" * 80)

else:

    print(
<<<<<<< Updated upstream
        "No embeddings available for similarity test."
=======
        "No newly generated embeddings available for "
        "the similarity test."
>>>>>>> Stashed changes
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete
# MAGIC
<<<<<<< Updated upstream
# MAGIC The Weather Intelligence embedding pipeline is now:
=======
# MAGIC The complete Weather Intelligence pipeline is:
>>>>>>> Stashed changes
# MAGIC
# MAGIC ```text
# MAGIC National Weather Service API
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/sync
# MAGIC             |
# MAGIC             v
# MAGIC        weather_news
# MAGIC             |
# MAGIC             v
# MAGIC ingest_weather_embeddings
# MAGIC             |
<<<<<<< Updated upstream
# MAGIC      narrative_text
# MAGIC             |
# MAGIC       chunking
=======
# MAGIC       narrative_text
>>>>>>> Stashed changes
# MAGIC             |
# MAGIC      800-char chunks
# MAGIC       100-char overlap
# MAGIC             |
# MAGIC             v
# MAGIC   all-MiniLM-L6-v2
# MAGIC        384 dimensions
# MAGIC             |
# MAGIC             v
# MAGIC     weather_embeddings
# MAGIC             |
# MAGIC       pgvector VECTOR(384)
# MAGIC             |
# MAGIC       HNSW / cosine
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/search
# MAGIC ```
# MAGIC
# MAGIC Example search request:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "query": "flash flood risk this weekend",
# MAGIC   "top_k": 5
# MAGIC }
# MAGIC ```
# MAGIC
<<<<<<< Updated upstream
# MAGIC The Flask API should embed the user's query using the same
# MAGIC `all-MiniLM-L6-v2` model and use pgvector's cosine-distance operator
# MAGIC (`<=>`) to retrieve the most semantically relevant weather chunks.
=======
# MAGIC The Flask API should return the most semantically relevant weather
# MAGIC <<<<<<< Updated upstream
# MAGIC chunks ranked by cosine similarity.
# MAGIC
# MAGIC ## Serverless Compatibility
# MAGIC
# MAGIC This notebook uses:
# MAGIC
# MAGIC - `pg8000` for PostgreSQL connectivity
# MAGIC - `sentence-transformers` for embeddings
# MAGIC - direct PostgreSQL inserts
# MAGIC - PostgreSQL `VECTOR(384)` casting
# MAGIC
# MAGIC It intentionally avoids:
# MAGIC
# MAGIC - `psycopg2-binary`
# MAGIC - Spark JDBC writes
# MAGIC - Spark writes to pgvector
# MAGIC - array-to-vector post-processing
# MAGIC =======
# MAGIC chunks ranked using cosine similarity.
# MAGIC >>>>>>> Stashed changes
>>>>>>> Stashed changes
