# pip install black isort ruff mypy
# # Formatea código
# black cubexpress/

# # Ordena imports
# isort cubexpress/

# # Lint
# ruff check --fix cubexpress/

# # Publica
# poetry version patch
# poetry build
# poetry publish


# # Commit los cambios de versión
# git add pyproject.toml
# git commit -m "chore: bump version to 0.1.20"

# # Tag la release
# git tag v0.1.20

# # Push todo
# git push origin main
# git push origin v0.1.20