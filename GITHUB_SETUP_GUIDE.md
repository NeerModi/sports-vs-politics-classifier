# GitHub Setup Instructions for Neer Modi

**Project:** Sports vs Politics News Classifier  
**GitHub Username:** neer-modi (or your actual GitHub username)  
**Repository Name:** sports-vs-politics-classifier  
**Status:** Ready to push

---

## Step-by-Step GitHub Setup Guide

### STEP 1: Create GitHub Repository (5 minutes)

#### Option A: Using GitHub Web Interface (Recommended)

1. **Go to GitHub.com**
   - Visit: https://github.com/new
   - Login with your GitHub account (create one if needed)

2. **Fill in Repository Details**
   - **Repository name:** `sports-vs-politics-classifier`
   - **Description:** Machine learning classifier for sports vs politics news using NLP
   - **Visibility:** Select **PUBLIC** (required for GitHub Pages)
   - Leave other options as default
   - Click **Create repository**

3. **Copy the Repository URL**
   - After creation, you'll see a URL like:
   - `https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git`
   - Copy this URL (you'll need it in Step 2)

#### Option B: Using GitHub CLI (if installed)

```bash
gh repo create sports-vs-politics-classifier --public --description "Machine learning classifier for sports vs politics news"
```

---

### STEP 2: Configure Git Locally

Open PowerShell in your project directory:

```bash
cd "C:\Users\neerm\Downloads\NLU-Assignment-1\B23CS1043_Problem4\sports-vs-politics-classifier"
```

#### 2a. Initialize Git (if not already done)

```bash
git init
```

#### 2b. Configure Your Git Identity

```bash
git config user.name "Neer Modi"
git config user.email "your.email@example.com"
```

Replace with your actual GitHub email address.

#### 2c. Add Remote Repository

```bash
git remote add origin https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git
```

Replace `YOUR_USERNAME` with your actual GitHub username.

**Verify it worked:**
```bash
git remote -v
```

Should show:
```
origin  https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git (fetch)
origin  https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git (push)
```

---

### STEP 3: Stage and Commit Files

```bash
# Add all files to git
git add .

# Verify what will be committed
git status

# Commit with message
git commit -m "Initial commit: Complete NLP sports vs politics classifier project with 97.32% accuracy"
```

**Optional - View commit log:**
```bash
git log
```

---

### STEP 4: Push to GitHub

#### First Push (Create main branch)

```bash
git branch -M main
git push -u origin main
```

This will:
1. Rename the default branch to `main`
2. Push all files to GitHub
3. Set up tracking for future pushes

**You may be prompted for authentication:**
- **Option A:** Enter GitHub username and personal access token
  - Generate token: https://github.com/settings/tokens
  - Select: `repo` scope
  
- **Option B:** Use GitHub CLI authentication
  ```bash
  gh auth login
  ```

#### Subsequent Pushes (after making changes)

```bash
git push origin main
```

---

### STEP 5: Enable GitHub Pages

1. **Go to Repository Settings**
   - Visit: `https://github.com/YOUR_USERNAME/sports-vs-politics-classifier/settings/pages`

2. **Configure GitHub Pages**
   - Under "Build and deployment"
   - **Source:** Select "Deploy from a branch"
   - **Branch:** Select `main`
   - **Folder:** Select `/ (root)` or `/docs`
   - Click **Save**

3. **Wait for Deployment**
   - GitHub will take 1-5 minutes to build
   - Check the Actions tab for build status
   - You'll see a green checkmark when complete

---

### STEP 6: Access Your GitHub Pages Website

Your site will be live at:

```
https://YOUR_USERNAME.github.io/sports-vs-politics-classifier/
```

**Individual Pages:**
- Home: `https://YOUR_USERNAME.github.io/sports-vs-politics-classifier/`
- Results: `https://YOUR_USERNAME.github.io/sports-vs-politics-classifier/results/`
- Models: `https://YOUR_USERNAME.github.io/sports-vs-politics-classifier/models/`
- Methodology: `https://YOUR_USERNAME.github.io/sports-vs-politics-classifier/methodology/`

---

## Complete Terminal Commands (Copy-Paste Ready)

```bash
# Navigate to project
cd "C:\Users\neerm\Downloads\NLU-Assignment-1\B23CS1043_Problem4\sports-vs-politics-classifier"

# Initialize git (if needed)
git init

# Configure identity
git config user.name "Neer Modi"
git config user.email "your.email@example.com"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git

# Stage all files
git add .

# Commit
git commit -m "Initial commit: Sports vs Politics Classifier - 97.32% accuracy with Linear SVM"

# Push to GitHub
git branch -M main
git push -u origin main
```

Replace:
- `your.email@example.com` with your GitHub email
- `YOUR_USERNAME` with your GitHub username

---

## Verification Checklist

After completing all steps, verify:

✓ **GitHub Repository**
  - [ ] Repository visible at https://github.com/YOUR_USERNAME/sports-vs-politics-classifier
  - [ ] Repository is PUBLIC
  - [ ] All files visible on GitHub

✓ **GitHub Pages Build**
  - [ ] Settings → Pages shows deployment status
  - [ ] Actions tab shows successful build (green checkmark)
  - [ ] Build took 1-5 minutes

✓ **Website Live**
  - [ ] Homepage loads at `https://YOUR_USERNAME.github.io/sports-vs-politics-classifier/`
  - [ ] All pages accessible (results, models, methodology)
  - [ ] Project metrics visible

✓ **Content Quality**
  - [ ] Report displays correctly
  - [ ] Images load properly
  - [ ] Navigation works
  - [ ] All links functional

---

## Troubleshooting Common Issues

### Issue: "remote origin already exists"

**Solution:** Remove and re-add
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/sports-vs-politics-classifier.git
```

### Issue: "Authentication failed"

**Solution:** Use personal access token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select `repo` scope
4. Copy the token
5. Use token as password when prompted

### Issue: GitHub Pages not showing content

**Solution:**
1. Check repository is PUBLIC (not private)
2. Verify `/docs` folder has `index.md`
3. Check `_config.yml` exists in root
4. Wait 5 minutes and refresh browser

### Issue: "branch main not found"

**Solution:** Ensure branch is created and pushed
```bash
git branch -M main
git push -u origin main
```

---

## GitHub Pages Features Available

Your published site includes:

✓ **Professional Theme** - jekyll-theme-minimal
✓ **Automatic HTTPS/SSL** - Secure connection
✓ **Mobile Responsive** - Works on all devices
✓ **SEO Optimized** - Search engine friendly
✓ **Auto RSS Feed** - Atom feed generation
✓ **Custom Domain** - Optional (domain.com)
✓ **Analytics** - Optional (Google Analytics)
✓ **CI/CD Ready** - GitHub Actions integration

---

## Making Future Updates

After initial setup, to update your site:

```bash
# Make changes to files
# Edit documentation, add new results, etc.

# Commit changes
git add .
git commit -m "Update: Added new analysis results"

# Push to GitHub
git push origin main
```

GitHub Pages will automatically rebuild within 1-5 minutes.

---

## Optional: Setup SSH Key (for password-less pushes)

If you want to avoid entering credentials repeatedly:

1. **Generate SSH Key**
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

2. **Add to GitHub**
   - Go to https://github.com/settings/keys
   - Click "New SSH key"
   - Paste your public key

3. **Update Remote URL**
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/sports-vs-politics-classifier.git
```

---

## Final Checklist

Before pushing, ensure you have:

- [ ] GitHub account created (https://github.com/signup)
- [ ] GitHub username ready
- [ ] Email address for git config
- [ ] Repository created on GitHub
- [ ] Project directory ready (current location)
- [ ] All files staged and ready to commit
- [ ] Repository README.md is updated
- [ ] /docs folder has all GitHub Pages files

---

## Success Indicators

You'll know everything worked when:

1. **GitHub shows your repository** with all files
2. **GitHub Pages build succeeds** (green checkmark in Actions)
3. **Your site is live** at the GitHub Pages URL
4. **All pages load correctly** with proper formatting
5. **Metrics and visualizations** display properly

---

## Support Resources

- **GitHub Docs:** https://docs.github.com
- **GitHub Pages Guide:** https://docs.github.com/en/pages
- **Git Tutorial:** https://git-scm.com/book/en/v2
- **Markdown Guide:** https://www.markdownguide.org

---

**Next Action:** Follow Steps 1-6 above to get your project live!

**Estimated Time:** 15-20 minutes total

**Result:** Professional GitHub portfolio showcasing your NLP project with 97.32% accuracy!
