# Release Procedures and Versioning

This guide covers the complete release process for NullStrike, including versioning strategy, quality assurance, and deployment procedures.

## Versioning Strategy

NullStrike follows [Semantic Versioning](https://semver.org/) (SemVer) with the format `MAJOR.MINOR.PATCH`:

- **MAJOR**: Breaking changes to public API or fundamental behavior
- **MINOR**: New features that are backward compatible  
- **PATCH**: Bug fixes and minor improvements

### Version Examples

```
1.0.0    - Initial stable release
1.1.0    - Added Fisher Information plugin system
1.1.1    - Fixed nullspace computation bug
1.2.0    - Added SBML model loading support
2.0.0    - Breaking change: New analysis result format
```

### Pre-release Versions

For development and testing:

```
1.2.0-alpha.1    - Early development version
1.2.0-beta.1     - Feature-complete, testing version
1.2.0-rc.1       - Release candidate
```

## Release Types

### Patch Releases (1.1.0 → 1.1.1)

**Criteria**: 
- Bug fixes only
- No new features
- No breaking changes
- Performance improvements
- Documentation updates

**Process**:
1. Fix identified bugs
2. Update tests if needed
3. Run full test suite
4. Update CHANGELOG.md
5. Tag and release

**Timeline**: As needed, typically within 1-2 weeks of bug identification

### Minor Releases (1.1.0 → 1.2.0)

**Criteria**:
- New features
- New analysis methods
- Enhanced visualizations
- Backward-compatible API changes
- Dependency updates

**Process**:
1. Feature development and testing
2. Documentation updates
3. Comprehensive testing
4. Beta testing period
5. Final release

**Timeline**: Every 2-3 months

### Major Releases (1.x.x → 2.0.0)

**Criteria**:
- Breaking API changes
- Major architectural changes
- Fundamental algorithm improvements
- Python version requirement changes

**Process**:
1. Extended development cycle
2. Migration guides
3. Alpha/beta testing period
4. Community feedback incorporation
5. Stable release

**Timeline**: Annually or as needed

## Release Workflow

### 1. Pre-Release Checklist

```bash
# Development branch preparation
git checkout develop
git pull origin develop

# Ensure all features are complete
git log --oneline main..develop

# Run comprehensive test suite
pytest --cov=nullstrike --cov-report=html
pytest tests/integration/ -v
pytest tests/performance/ -m "not stress"

# Check code quality
black --check src/ tests/
flake8 src/ tests/
mypy src/

# Verify examples work
nullstrike C2M
nullstrike Bolie  
nullstrike calibration_single

# Check documentation builds
mkdocs build --strict
```

### 2. Version Update Process

```python
# scripts/update_version.py
"""Script to update version across all files."""

import re
import sys
from pathlib import Path

def update_version(new_version: str):
    """Update version in all relevant files."""
    
    files_to_update = [
        'pyproject.toml',
        'src/nullstrike/__init__.py',
        'docs/index.md',
        'CITATION.cff'
    ]
    
    for file_path in files_to_update:
        update_version_in_file(file_path, new_version)
    
    print(f"Updated version to {new_version} in {len(files_to_update)} files")

def update_version_in_file(file_path: str, new_version: str):
    """Update version in a specific file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Warning: {file_path} not found")
        return
    
    content = path.read_text()
    
    # Different patterns for different files
    if file_path.endswith('pyproject.toml'):
        pattern = r'version = "[^"]*"'
        replacement = f'version = "{new_version}"'
    elif file_path.endswith('__init__.py'):
        pattern = r'__version__ = "[^"]*"'
        replacement = f'__version__ = "{new_version}"'
    elif file_path.endswith('.md'):
        pattern = r'Version [0-9]+\.[0-9]+\.[0-9]+'
        replacement = f'Version {new_version}'
    elif file_path.endswith('.cff'):
        pattern = r'version: "[^"]*"'
        replacement = f'version: "{new_version}"'
    else:
        print(f"Unknown file type: {file_path}")
        return
    
    updated_content = re.sub(pattern, replacement, content)
    path.write_text(updated_content)
    print(f"Updated {file_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$', new_version):
        print("Invalid version format. Use semantic versioning (e.g., 1.2.3)")
        sys.exit(1)
    
    update_version(new_version)
```

### 3. Changelog Management

```markdown
# CHANGELOG.md Template

# Changelog

All notable changes to NullStrike will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features that have been added

### Changed  
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Features that have been removed

### Fixed
- Bug fixes

### Security
- Security improvements

## [1.2.0] - 2024-03-15

### Added
- Fisher Information Matrix analysis plugin
- SBML model import support
- Enhanced visualization customization options
- Parallel computation support for large models

### Changed
- Improved nullspace computation algorithm for better numerical stability
- Updated CLI interface with better progress indicators
- Enhanced error messages with actionable suggestions

### Fixed
- Fixed memory leak in visualization generation
- Corrected parameter combination identification edge cases
- Fixed compatibility issues with SymPy 1.12

### Performance
- 3x faster nullspace computation for models with >20 parameters
- Reduced memory usage by 40% for large models
- Optimized visualization rendering

## [1.1.1] - 2024-02-20

### Fixed
- Critical bug in nullspace dimension calculation
- Incorrect parameter combination ordering in some cases
- Visualization crashes with certain parameter ranges

### Security
- Updated dependencies to address security vulnerabilities

## [1.1.0] - 2024-01-10

### Added
- Plugin system for custom analysis methods
- Advanced checkpointing for long computations
- Batch analysis capabilities
- Integration with external tools (MATLAB, R)

### Changed
- Improved documentation with more examples
- Better error handling and user feedback

### Fixed
- Various minor bugs and edge cases

## [1.0.0] - 2023-12-01

### Added
- Initial stable release
- Core STRIKE-GOLDD implementation
- Nullspace analysis for parameter combinations
- 3D visualization system
- Command-line interface
- Python API
- Comprehensive documentation

[Unreleased]: https://github.com/vipulsinghal02/NullStrike/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/vipulsinghal02/NullStrike/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/vipulsinghal02/NullStrike/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/vipulsinghal02/NullStrike/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/vipulsinghal02/NullStrike/releases/tag/v1.0.0
```

### 4. Release Branch Creation

```bash
#!/bin/bash
# scripts/create_release_branch.sh

set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "Creating release branch for version $VERSION"

# Ensure we're on develop and up to date
git checkout develop
git pull origin develop

# Create release branch
RELEASE_BRANCH="release/v$VERSION"
git checkout -b "$RELEASE_BRANCH"

# Update version numbers
python scripts/update_version.py "$VERSION"

# Update changelog
echo "Please update CHANGELOG.md with release notes for version $VERSION"
echo "Press Enter when ready to continue..."
read

# Commit version updates
git add .
git commit -m "chore: bump version to $VERSION"

# Push release branch
git push origin "$RELEASE_BRANCH"

echo "Release branch $RELEASE_BRANCH created and pushed"
echo "Next steps:"
echo "1. Review and finalize CHANGELOG.md"
echo "2. Run final testing"
echo "3. Create pull request to main"
echo "4. After merge, tag the release"
```

### 5. Quality Assurance Process

```python
# scripts/qa_checklist.py
"""Quality assurance checklist for releases."""

import subprocess
import sys
from pathlib import Path

class QAChecker:
    """Quality assurance checker for releases."""
    
    def __init__(self):
        self.passed_checks = []
        self.failed_checks = []
    
    def run_all_checks(self):
        """Run complete QA checklist."""
        print("Running NullStrike QA Checklist")
        print("=" * 40)
        
        checks = [
            self.check_version_consistency,
            self.check_code_quality,
            self.check_test_coverage,
            self.check_documentation,
            self.check_examples,
            self.check_performance_regression,
            self.check_dependencies,
            self.check_security
        ]
        
        for check in checks:
            try:
                check()
                self.passed_checks.append(check.__name__)
                print(f"PASS: {check.__name__}")
            except Exception as e:
                self.failed_checks.append((check.__name__, str(e)))
                print(f"FAIL: {check.__name__}: {e}")
        
        self.print_summary()
        
        return len(self.failed_checks) == 0
    
    def check_version_consistency(self):
        """Check version consistency across files."""
        version_files = [
            'pyproject.toml',
            'src/nullstrike/__init__.py'
        ]
        
        versions = []
        for file_path in version_files:
            version = self._extract_version(file_path)
            versions.append(version)
        
        if len(set(versions)) != 1:
            raise AssertionError(f"Version mismatch: {dict(zip(version_files, versions))}")
    
    def check_code_quality(self):
        """Check code quality with linting tools."""
        # Black formatting
        result = subprocess.run(['black', '--check', 'src/', 'tests/'], 
                              capture_output=True)
        if result.returncode != 0:
            raise AssertionError("Code formatting issues found (run: black src/ tests/)")
        
        # Flake8 linting
        result = subprocess.run(['flake8', 'src/', 'tests/'], 
                              capture_output=True)
        if result.returncode != 0:
            raise AssertionError("Linting issues found")
        
        # Type checking
        result = subprocess.run(['mypy', 'src/'], 
                              capture_output=True)
        if result.returncode != 0:
            raise AssertionError("Type checking issues found")
    
    def check_test_coverage(self):
        """Check test coverage meets requirements."""
        result = subprocess.run([
            'pytest', '--cov=nullstrike', '--cov-report=term-missing',
            '--cov-fail-under=80'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise AssertionError("Test coverage below 80% or tests failing")
    
    def check_documentation(self):
        """Check documentation builds correctly."""
        result = subprocess.run(['mkdocs', 'build', '--strict'], 
                              capture_output=True)
        if result.returncode != 0:
            raise AssertionError("Documentation build failed")
    
    def check_examples(self):
        """Check that examples run successfully."""
        examples = ['C2M', 'Bolie', 'calibration_single']
        
        for example in examples:
            result = subprocess.run([
                'nullstrike', example, '--parameters-only'
            ], capture_output=True, timeout=300)
            
            if result.returncode != 0:
                raise AssertionError(f"Example {example} failed")
    
    def check_performance_regression(self):
        """Check for performance regressions."""
        # Run performance benchmarks
        result = subprocess.run([
            'pytest', 'tests/performance/', '-m', 'benchmark'
        ], capture_output=True)
        
        if result.returncode != 0:
            raise AssertionError("Performance benchmarks failed")
    
    def check_dependencies(self):
        """Check dependency versions and security."""
        # Check for dependency conflicts
        result = subprocess.run(['pip', 'check'], capture_output=True)
        if result.returncode != 0:
            raise AssertionError("Dependency conflicts detected")
    
    def check_security(self):
        """Check for security vulnerabilities."""
        try:
            result = subprocess.run(['safety', 'check'], capture_output=True)
            if result.returncode != 0:
                raise AssertionError("Security vulnerabilities found")
        except FileNotFoundError:
            print("Warning: 'safety' not installed, skipping security check")
    
    def _extract_version(self, file_path):
        """Extract version from file."""
        content = Path(file_path).read_text()
        
        if 'pyproject.toml' in file_path:
            import re
            match = re.search(r'version = "([^"]*)"', content)
            return match.group(1) if match else None
        elif '__init__.py' in file_path:
            import re
            match = re.search(r'__version__ = "([^"]*)"', content)
            return match.group(1) if match else None
    
    def print_summary(self):
        """Print QA summary."""
        print("\nQA Summary:")
        print("=" * 20)
        print(f"Passed: {len(self.passed_checks)}")
        print(f"Failed: {len(self.failed_checks)}")
        
        if self.failed_checks:
            print("\nFailed checks:")
            for check, error in self.failed_checks:
                print(f"  {check}: {error}")

if __name__ == '__main__':
    qa = QAChecker()
    success = qa.run_all_checks()
    sys.exit(0 if success else 1)
```

### 6. Release Finalization

```bash
#!/bin/bash
# scripts/finalize_release.sh

set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "Finalizing release $VERSION"

# Ensure release branch exists and is current
RELEASE_BRANCH="release/v$VERSION"
git checkout "$RELEASE_BRANCH"
git pull origin "$RELEASE_BRANCH"

# Run final QA checks
echo "Running final QA checks..."
python scripts/qa_checklist.py

if [ $? -ne 0 ]; then
    echo "QA checks failed. Please fix issues before release."
    exit 1
fi

# Merge to main
echo "Merging to main..."
git checkout main
git pull origin main
git merge --no-ff "$RELEASE_BRANCH" -m "Release version $VERSION"

# Create and push tag
echo "Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"
git push origin main
git push origin "v$VERSION"

# Merge back to develop
echo "Merging back to develop..."
git checkout develop
git pull origin develop
git merge --no-ff main -m "Merge release $VERSION back to develop"
git push origin develop

# Clean up release branch
echo "Cleaning up release branch..."
git branch -d "$RELEASE_BRANCH"
git push origin --delete "$RELEASE_BRANCH"

echo "Release $VERSION completed successfully!"
echo "Next steps:"
echo "1. Monitor GitHub Actions for build/test results"
echo "2. Create GitHub release with release notes"
echo "3. Update documentation if needed"
echo "4. Announce release to community"
```

## GitHub Release Process

### 1. Automated Release Creation

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=nullstrike
    
    - name: Build package
      run: |
        python -m build
    
    - name: Generate release notes
      id: release_notes
      run: |
        python scripts/generate_release_notes.py ${{ github.ref_name }}
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        body_path: release_notes.md
        files: |
          dist/*.whl
          dist/*.tar.gz
        draft: false
        prerelease: ${{ contains(github.ref, '-') }}
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Publish to PyPI
      if: startsWith(github.ref, 'refs/tags/v') && !contains(github.ref, '-')
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
```

### 2. Release Notes Generation

```python
# scripts/generate_release_notes.py
"""Generate release notes from changelog and git history."""

import sys
import re
from pathlib import Path
import subprocess

def generate_release_notes(version: str):
    """Generate release notes for a specific version."""
    
    # Extract changelog section
    changelog_section = extract_changelog_section(version)
    
    # Get git statistics
    git_stats = get_git_statistics(version)
    
    # Generate contributors list
    contributors = get_contributors(version)
    
    # Combine into release notes
    release_notes = format_release_notes(
        version, changelog_section, git_stats, contributors
    )
    
    # Write to file
    with open('release_notes.md', 'w') as f:
        f.write(release_notes)
    
    print(f"Release notes generated for {version}")

def extract_changelog_section(version: str) -> str:
    """Extract the relevant section from CHANGELOG.md."""
    changelog = Path('CHANGELOG.md').read_text()
    
    # Find version section
    pattern = rf'## \[{re.escape(version)}\].*?(?=## \[|$)'
    match = re.search(pattern, changelog, re.DOTALL)
    
    if match:
        return match.group(0).strip()
    else:
        return f"No changelog entry found for {version}"

def get_git_statistics(version: str) -> dict:
    """Get git statistics for the release."""
    # Get previous tag
    result = subprocess.run([
        'git', 'describe', '--tags', '--abbrev=0', f'{version}^'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        prev_tag = result.stdout.strip()
    else:
        # If no previous tag, use first commit
        prev_tag = subprocess.run([
            'git', 'rev-list', '--max-parents=0', 'HEAD'
        ], capture_output=True, text=True).stdout.strip()
    
    # Get commit count
    commit_count = subprocess.run([
        'git', 'rev-list', '--count', f'{prev_tag}..{version}'
    ], capture_output=True, text=True).stdout.strip()
    
    # Get file changes
    files_changed = subprocess.run([
        'git', 'diff', '--name-only', f'{prev_tag}..{version}'
    ], capture_output=True, text=True).stdout.strip().split('\n')
    
    return {
        'commits': int(commit_count),
        'files_changed': len([f for f in files_changed if f]),
        'previous_tag': prev_tag
    }

def get_contributors(version: str) -> list:
    """Get list of contributors for the release."""
    result = subprocess.run([
        'git', 'shortlog', '-sn', f'{version}^..{version}'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        contributors = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                # Extract name (ignore commit count)
                name = line.strip().split('\t')[1]
                contributors.append(name)
        return contributors
    else:
        return []

def format_release_notes(version: str, changelog: str, stats: dict, contributors: list) -> str:
    """Format complete release notes."""
    notes = [f"# NullStrike {version}"]
    notes.append("")
    
    # Add changelog content
    notes.append(changelog)
    notes.append("")
    
    # Add statistics
    notes.append("## Release Statistics")
    notes.append(f"- **Commits**: {stats['commits']}")
    notes.append(f"- **Files changed**: {stats['files_changed']}")
    notes.append("")
    
    # Add contributors
    if contributors:
        notes.append("## Contributors")
        notes.append("Thanks to the following contributors:")
        for contributor in contributors:
            notes.append(f"- {contributor}")
        notes.append("")
    
    # Add installation instructions
    notes.append("## Installation")
    notes.append("```bash")
    notes.append(f"pip install nullstrike=={version}")
    notes.append("```")
    notes.append("")
    
    # Add verification
    notes.append("## Verification")
    notes.append("```bash")
    notes.append("nullstrike --version")
    notes.append("nullstrike C2M --parameters-only")
    notes.append("```")
    
    return "\n".join(notes)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_release_notes.py <version>")
        sys.exit(1)
    
    version = sys.argv[1].lstrip('v')  # Remove 'v' prefix if present
    generate_release_notes(version)
```

## Post-Release Tasks

### 1. Release Announcement

```markdown
# Release Announcement Template

Subject: NullStrike v1.2.0 Released - New Features and Performance Improvements

Dear NullStrike Community,

We're excited to announce the release of NullStrike v1.2.0! This release includes significant new features, performance improvements, and bug fixes.

## What's New

### Major Features
- Fisher Information Matrix analysis plugin
- SBML model import support  
- Enhanced visualization customization
- Parallel computation for large models

### Performance Improvements
- 3x faster nullspace computation for models with >20 parameters
- 40% reduction in memory usage for large models
- Optimized visualization rendering

### Bug Fixes
- Fixed critical nullspace dimension calculation bug
- Corrected parameter combination identification edge cases
- Fixed compatibility issues with SymPy 1.12

## Installation

```bash
pip install --upgrade nullstrike
```

## Verification

```bash
nullstrike --version
nullstrike C2M --parameters-only
```

## Documentation

Updated documentation is available at: https://vipulsinghal02.github.io/NullStrike/

## Contributors

Special thanks to all contributors who made this release possible:
- [List of contributors]

## Reporting Issues

If you encounter any issues, please report them at:
https://github.com/vipulsinghal02/NullStrike/issues

Happy analyzing!
The NullStrike Team
```

### 2. Documentation Updates

```bash
#!/bin/bash
# scripts/post_release_docs.sh

VERSION=$1

echo "Updating documentation for release $VERSION"

# Update version in documentation
sed -i "s/Version [0-9]\+\.[0-9]\+\.[0-9]\+/Version $VERSION/g" docs/index.md

# Rebuild and deploy documentation
mkdocs build
mkdocs gh-deploy

echo "Documentation updated and deployed"
```

### 3. Community Updates

```python
# scripts/update_community.py
"""Update community resources after release."""

import json
import requests
from pathlib import Path

def update_conda_forge():
    """Update conda-forge recipe (manual process)."""
    print("Manual steps for conda-forge update:")
    print("1. Fork conda-forge/nullstrike-feedstock")
    print("2. Update recipe/meta.yaml with new version and sha256")
    print("3. Create pull request")
    print("4. Wait for CI and merge")

def update_zenodo():
    """Update Zenodo record."""
    print("Manual steps for Zenodo update:")
    print("1. Go to https://zenodo.org/account/settings/github/")
    print("2. Find NullStrike repository")
    print("3. Create new version")
    print("4. Update metadata if needed")

def check_pypi_status(package_name="nullstrike"):
    """Check PyPI package status."""
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    
    if response.status_code == 200:
        data = response.json()
        latest_version = data['info']['version']
        print(f"Latest version on PyPI: {latest_version}")
        return latest_version
    else:
        print("Failed to check PyPI status")
        return None

if __name__ == '__main__':
    print("Post-release community updates")
    print("=" * 30)
    
    # Check PyPI
    pypi_version = check_pypi_status()
    
    # Update conda-forge
    update_conda_forge()
    
    # Update Zenodo
    update_zenodo()
```

## Release Calendar

### Planned Release Schedule

```
2024 Release Calendar:
- January: v1.1.0 (Minor - Plugin system)
- February: v1.1.1 (Patch - Bug fixes)  
- April: v1.2.0 (Minor - SBML support)
- July: v1.3.0 (Minor - Performance improvements)
- October: v2.0.0 (Major - API restructure)

Patch releases: As needed for critical bugs
Security releases: Immediate for vulnerabilities
```

### Release Decision Matrix

| Issue Type | Patch | Minor | Major |
|------------|-------|-------|-------|
| Bug fix | Yes | Yes | Yes |
| New feature | No | Yes | Yes |
| Performance improvement | Yes | Yes | Yes |
| API change (backward compatible) | No | Yes | Yes |
| Breaking change | No | No | Yes |
| Documentation | Yes | Yes | Yes |
| Dependency update (major) | No | Yes | Yes |
| Security fix | Yes | Yes | Yes |

This comprehensive release process ensures that NullStrike releases are high-quality, well-documented, and properly communicated to the community. The combination of automated testing, manual QA checks, and systematic versioning provides confidence in each release.

---

## Next Steps

1. **Study the current release process** by examining recent releases
2. **Practice the workflow** on a development branch
3. **Contribute to release planning** through GitHub discussions
4. **Help with QA testing** during pre-release phases
5. **Explore the [Advanced Features](../advanced/batch.md)** for power users