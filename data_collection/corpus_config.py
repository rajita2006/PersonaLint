"""
Corpus configuration for PersonaLint.

Each entry: (repo_dir_name, author_name). Bot accounts (dependabot,
pre-commit-ci, etc.) are excluded -- we want real individual coding style.
"""

CORPUS = [
    ("requests", "Nate Prewitt"),
    ("requests", "Ian Stapleton Cordasco"),
    ("flask", "David Lord"),
    ("httpx", "Tom Christie"),
    ("httpx", "Kar Petrosyan"),
    ("black", "cobalt"),
    ("click", "David Lord"),
    ("click", "Kevin Deldycke"),
    ("gunicorn", "Benoit Chesneau"),
    ("gunicorn", "Randall Leeds"),
    ("urllib3", "Illia Volochii"),
    ("urllib3", "Quentin Pradet"),
    ("attrs", "Hynek Schlawack"),
    ("tornado", "Ben Darnell"),
    ("tornado", "Pierce Lopez"),
    ("celery", "Asif Saif Uddin"),
]

MAX_COMMITS_PER_AUTHOR = 80
