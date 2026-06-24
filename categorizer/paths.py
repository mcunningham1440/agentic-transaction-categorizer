"""Canonical filesystem locations for the project.

Every module resolves config/data/output paths through here so that moving the
package or running from a different cwd doesn't break path resolution. All paths
are anchored to PROJECT_ROOT (the directory that contains the categorizer/
package), not to each module's own location.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Hand-edited / secret inputs (all gitignored except the *.sample templates).
ENV_PATH = os.path.join(CONFIG_DIR, ".env")
CATEGORIES_PATH = os.path.join(CONFIG_DIR, "categories.yaml")
CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
TOKEN_PATH = os.path.join(CONFIG_DIR, "token.json")
