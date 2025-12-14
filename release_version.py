#!/usr/bin/env python3
"""Script to update version and create GitHub tag/release for beamz."""

import re
import sys
import os
import subprocess
import argparse
from pathlib import Path

def update_version_in_file(filepath, version, pattern, replacement_template):
    """Update version in a file using regex pattern."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist, skipping")
        return False
    
    content = filepath.read_text()
    new_content = re.sub(pattern, replacement_template.format(version=version), content)
    
    if content != new_content:
        filepath.write_text(new_content)
        print(f"Updated version in {filepath}")
        return True
    else:
        print(f"No change needed in {filepath}")
        return False

def update_version(version):
    """Update version in all relevant files."""
    changes = []
    
    # Update setup.py
    changes.append(update_version_in_file(
        "setup.py",
        version,
        r'version="[^"]+"',
        'version="{version}"'
    ))
    
    # Update pyproject.toml
    changes.append(update_version_in_file(
        "pyproject.toml",
        version,
        r'version = "[^"]+"',
        'version = "{version}"'
    ))
    
    # Update beamz/__init__.py
    changes.append(update_version_in_file(
        "beamz/__init__.py",
        version,
        r'__version__ = "[^"]+"',
        '__version__ = "{version}"'
    ))
    
    return any(changes)

def validate_version(version):
    """Validate version string format (semantic versioning)."""
    pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
    if not re.match(pattern, version):
        raise ValueError(f"Invalid version format: {version}. Expected format: X.Y.Z or X.Y.Z-suffix")
    return version

def get_current_branch():
    """Get current git branch."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def check_git_status():
    """Check if git working directory is clean."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip() == ""

def create_git_tag(version, message=None):
    """Create a git tag for the version."""
    if message is None:
        message = f"Release version {version}"
    
    # Check if tag already exists
    result = subprocess.run(
        ["git", "tag", "-l", f"v{version}"],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        print(f"Tag v{version} already exists. Use --force to overwrite.")
        return False
    
    subprocess.run(
        ["git", "tag", "-a", f"v{version}", "-m", message],
        check=True
    )
    print(f"Created git tag: v{version}")
    return True

def push_tag(version, remote="origin"):
    """Push tag to remote repository."""
    subprocess.run(
        ["git", "push", remote, f"v{version}"],
        check=True
    )
    print(f"Pushed tag v{version} to {remote}")

def create_github_release(version, token=None, message=None, draft=False):
    """Create a GitHub release using GitHub API."""
    if token is None:
        print("GitHub token not provided. Skipping GitHub release creation.")
        print("Set GITHUB_TOKEN environment variable or use --github-token flag.")
        return False
    
    if message is None:
        message = f"Release version {version}"
    
    import json
    import urllib.request
    import urllib.error
    
    repo = "QuentinWach/beamz"  # From setup.py
    url = f"https://api.github.com/repos/{repo}/releases"
    
    data = {
        "tag_name": f"v{version}",
        "name": f"v{version}",
        "body": message,
        "draft": draft
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode())
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            print(f"Created GitHub release: {result['html_url']}")
            return True
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"Error creating GitHub release: {e.code} - {error_body}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Update version and create GitHub tag/release for beamz"
    )
    parser.add_argument(
        "version",
        type=validate_version,
        help="Version string (e.g., 0.1.6)"
    )
    parser.add_argument(
        "--message", "-m",
        help="Release message (default: 'Release version X.Y.Z')"
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="Only create git tag, don't push or create GitHub release"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push tag to remote"
    )
    parser.add_argument(
        "--github-token",
        help="GitHub personal access token for creating releases"
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Create draft GitHub release"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing tag"
    )
    parser.add_argument(
        "--skip-version-update",
        action="store_true",
        help="Skip updating version files (use existing version)"
    )
    
    args = parser.parse_args()
    
    # Check we're on main branch
    branch = get_current_branch()
    if branch != "main":
        print(f"Warning: Not on main branch (currently on {branch})")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    # Check git status
    if not check_git_status():
        print("Warning: Working directory is not clean. Uncommitted changes detected.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    # Update version files
    if not args.skip_version_update:
        if not update_version(args.version):
            print("No version files were updated.")
            response = input("Continue with tag creation? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
    
    # Create git tag
    if args.force:
        # Delete existing tag if it exists
        subprocess.run(
            ["git", "tag", "-d", f"v{args.version}"],
            capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "--delete", f"v{args.version}"],
            capture_output=True
        )
    
    if not create_git_tag(args.version, args.message):
        if not args.force:
            print("Tag creation failed. Use --force to overwrite existing tag.")
            sys.exit(1)
        create_git_tag(args.version, args.message)
    
    if args.tag_only:
        print("Tag-only mode: skipping push and GitHub release")
        return
    
    # Push tag
    if not args.no_push:
        try:
            push_tag(args.version)
        except subprocess.CalledProcessError as e:
            print(f"Error pushing tag: {e}")
            sys.exit(1)
    
    # Create GitHub release
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if token:
        create_github_release(args.version, token, args.message, args.draft)
    else:
        print("\nTo create a GitHub release, set GITHUB_TOKEN environment variable or use --github-token")
    
    print(f"\n✓ Version {args.version} released successfully!")
    print(f"  - Version files updated")
    print(f"  - Git tag v{args.version} created")
    if not args.no_push:
        print(f"  - Tag pushed to remote")

if __name__ == "__main__":
    import os
    main()

