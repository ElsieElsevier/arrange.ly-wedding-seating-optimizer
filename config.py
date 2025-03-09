import os

# Flask App Configuration
DEBUG = True
SECRET_KEY = os.environ.get('SECRET_KEY', 'development-secret-key')

# API Base URL
API_BASE_URL = '/api'

# Default optimization values
DEFAULT_NUM_TABLES = 10
DEFAULT_CAPACITY_PER_TABLE = 8

# Storage settings
# For production, you would configure database connections here
IN_MEMORY_STORAGE = True