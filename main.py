import requests
import sqlalchemy as sa

from pipeline.process import api_call
from pipeline.ragStep import rag_step
from secret.config import OPENAI_KEY, ANTHROPIC_KEY, GOOGLE_KEY, SQL_URL

# Setup params for query engine
params = requests.get(SQL_URL).json()['params']
engine = sa.create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}")
engine.begin()

# TODO: Add the entire pipeline here
