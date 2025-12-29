# common/schemas.py

"""
Canonical MARVA Requirement Schema

{
  "req_id": str,            # Global unique MARVA ID
  "text": str,              # Requirement text (verbatim)
  "source": str,            # Dataset name (qure, promised, etc.)
  "group_id": None,         # Always None in Step 1.2
  "metadata": {
      "original_id": str    # ID from the original dataset
  }
}
"""
