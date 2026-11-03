import os

# Flask App Configuration
# Debug mode exposes an interactive debugger, so it must be turned on explicitly.
DEBUG = os.environ.get('FLASK_DEBUG') == '1'
SECRET_KEY = os.environ.get('SECRET_KEY', 'development-secret-key')

# API Base URL
API_BASE_URL = '/api'

# Default optimization values
DEFAULT_NUM_TABLES = 10
DEFAULT_CAPACITY_PER_TABLE = 8

# Storage settings
# For production, you would configure database connections here
IN_MEMORY_STORAGE = True