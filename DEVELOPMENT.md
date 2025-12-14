## Development Guide / Notes



## Documentation
To test out the documentation locally, do:
```bash
mkdocs serve
```
To deploy it, type:
```bash
mkdocs gh-deploy
```
which will then create all the needed files on the gh-deploy branch and, well, deploy it there as a github-page.


## Version Release

To create a new version release and GitHub tag:

```bash
python release_version.py 0.1.6
```

This will:
1. Update version in `setup.py`, `pyproject.toml`, and `beamz/__init__.py`
2. Create a git tag `v0.1.6`
3. Push the tag to the remote repository

To also create a GitHub release (requires GitHub token):

```bash
export GITHUB_TOKEN=your_token_here
python scripts/release_version.py 0.1.6 --message "Release notes here"
```

Or pass the token directly:

```bash
python scripts/release_version.py 0.1.6 --github-token your_token_here
```

Additional options:
- `--tag-only`: Only create git tag, don't push or create GitHub release
- `--no-push`: Don't push tag to remote
- `--draft`: Create draft GitHub release
- `--force`: Force overwrite existing tag
- `--skip-version-update`: Skip updating version files

## Package Publishing
First update the version numbers in the `setup.py` file and others! Then
```bash
python -m build
```
then
```bash
python patch_wheel.py
```
then
```bash
python -m twine upload dist/beamz-0.1.0-py3-none-any.whl   
```
(though with the correct version) in order to publis the newest version of the package to pypi.