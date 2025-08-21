# GitHub Pages Setup and Troubleshooting

This document provides comprehensive instructions for setting up and troubleshooting GitHub Pages deployment for the NullStrike documentation.

## Repository Settings Configuration

### 1. Enable GitHub Pages

1. Navigate to your repository: https://github.com/vipulsinghal02/NullStrike
2. Go to **Settings** → **Pages** (in the left sidebar)
3. Under "Source", select **GitHub Actions**
4. This enables GitHub Actions to deploy to Pages

### 2. Repository Permissions

If your repository is under an organization, ensure:

1. Go to **Settings** → **Actions** → **General**
2. Under "Workflow permissions", select:
   - **Read and write permissions** (recommended)
   - OR **Read repository contents and packages permissions** with "Allow GitHub Actions to create and approve pull requests" checked

### 3. Pages Environment (Optional but Recommended)

1. Go to **Settings** → **Environments**
2. Create an environment named `github-pages` if it doesn't exist
3. Add any required reviewers or protection rules

## Troubleshooting Common Errors

### Error: "Resource not accessible by integration"

This error typically occurs when:

**Cause 1: Pages not enabled in repository settings**
- Solution: Follow step 1 above to enable GitHub Pages with GitHub Actions as the source

**Cause 2: Insufficient token permissions**
- Solution: The updated workflow now includes `actions: read` permission and better error handling

**Cause 3: Organization restrictions**
- Solution: Contact your organization admin to enable Pages for the repository

**Cause 4: Repository is private with free GitHub account**
- Solution: GitHub Pages on private repositories requires GitHub Pro or organization plan

### Error: "Get Pages site failed"

This indicates the repository doesn't have Pages configured yet:
- The updated workflow now handles this with `continue-on-error: true` and provides a fallback URL

### Error: "Deploy to GitHub Pages failed"

Common solutions:
1. Ensure the `site/` directory contains valid HTML files
2. Check that `index.html` exists in the site root
3. Verify MkDocs build completed successfully

## Updated Workflow Features

The updated `.github/workflows/docs.yml` includes:

1. **Enhanced Permissions**: Added `actions: read` for better integration
2. **Error Handling**: Pages setup continues even if configuration fails
3. **Fallback URL**: Automatically generates repository-based URL if Pages setup fails
4. **Dynamic site_url**: Updates MkDocs configuration with correct base URL

## Manual Verification Steps

After updating the workflow:

1. **Check Repository Settings**:
   ```bash
   # Verify Pages is enabled via GitHub CLI (if available)
   gh api repos/vipulsinghal02/NullStrike/pages
   ```

2. **Test Workflow Locally** (optional):
   ```bash
   # Build documentation locally to verify it works
   pip install -r requirements-docs.txt
   mkdocs build --clean --verbose
   ```

3. **Monitor Workflow Execution**:
   - Go to **Actions** tab in your repository
   - Watch the workflow execution for any remaining errors
   - Check the logs for each step

## Alternative Deployment Method

If GitHub Actions deployment continues to fail, you can use the traditional approach:

1. Build documentation locally:
   ```bash
   mkdocs gh-deploy --force
   ```

2. Or use a simpler workflow that pushes to `gh-pages` branch:
   - This method doesn't require special Pages permissions
   - Uses the traditional `gh-pages` branch approach

## Expected Outcomes

After successful setup:
- Documentation will be available at: `https://vipulsinghal02.github.io/NullStrike/`
- Workflow will run on every push to main branch
- Build artifacts will be automatically deployed to GitHub Pages

## Support Information

If issues persist after following these steps:

1. Check the Actions logs for specific error messages
2. Verify your GitHub account/organization has Pages enabled
3. Consider reaching out to GitHub Support if the issue appears to be platform-related

The updated workflow is designed to be more resilient to common permissions issues while maintaining full functionality when permissions are properly configured.