.PHONY: clean fix auto

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
fix:
	black src mp_lib scripts
auto:
	autoflake -i -r --remove-all-unused-imports src mp_lib scripts
