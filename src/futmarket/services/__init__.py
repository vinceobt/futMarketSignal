"""Service layer: the callable core shared by the CLI and the web API.

Keeping logic here (rather than in cli.py handlers or webapp.py routes) means
both interfaces call the same functions, and the background job runner can
invoke them off the request thread.
"""
