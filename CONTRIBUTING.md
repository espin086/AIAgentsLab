# Contributing to This Project

Thank you for your interest in contributing! We value your support and are excited to collaborate with you. Please read the following guidelines to ensure a smooth and efficient contribution process.

---

## Contribution Guidelines

### 1. Fork and Clone the Repository

**NOTE: You are cannot commit directly to main branch, only pull requests are accepted and enforced**

1. Fork the repository to your own GitHub account.
2. Clone your forked repository to your local machine:
```bash
git clone reporurl.git
```

3. Install pre-commit hooks
```bash
pip install pre-commit
pre-commit install
```


4. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```
---

### 3. Starting a New Project

If you are starting a new project within the monorepo:
	1.	Copy the Template Project:
	- Duplicate the project-name directory into a new folder named after your project:

```bash
cp -r project_name new_project_name
```

Update the new project’s README.md, requirements.txt, and other relevant files to reflect your new project.
All your work for the new project should stay within its designated directory to maintain modularity.
Add a brief description of your project to the list of projects in the monorepo-level README.md.

### 2. Code Formatting with Black

This project enforces Python code formatting using **Black**, an opinionated code formatter. Proper code formatting is required for all contributions to maintain code consistency.

- **Pre-commit Hook**: A pre-commit hook is configured to automatically format code using Black. If your code is not properly formatted, the commit will fail.
- **GitHub Action**: We use a GitHub Action to verify code formatting. Any code that is not formatted with Black will cause the CI pipeline to fail.

#### **How to Format Code with Black**

#### **For Python Files:**

1. Install Black:
   ```bash
   pip install black
   ```

2. Format your code locally before committing:
   ```bash
   black .
   ```

#### **For Jupyter Notebooks:**

1. Install Black:
   ```bash
   pip install nbqa black
   ```

2. Format your code locally before committing:
   ```bash
   nbqa black your_notebook.ipynb
   ```

If you attempt to commit unformatted code:
   - The pre-commit hook will automatically format the code for you.
   - You will need to recommit the changes after they have been formatted.

#### **(Optional but Useful) Automatic Black Formatting in VS Code**

You can integrate Black with Visual Studio Code for automatic formatting.

1. Install the **Black Formatter Extension** in VS Code.
2. Update your VS Code settings (View > Command Palette... and run Preferences: Open User Settings (JSON)):

   ```json
   "[python]": {
       "editor.defaultFormatter": "ms-python.black-formatter"
   }
   ```

3. To enable format-on-save for Python files, add this to your settings:
   ```json
   "[python]": {
       "editor.defaultFormatter": "ms-python.black-formatter",
       "editor.formatOnSave": true
   }
   ```

---

### 3. Linting Python Files and Jupyter Notebooks

#### **Linting Python Files**
Before committing your code, ensure it passes all checks using Pylint for Python code quality:
1. Install Pylint:
   ```bash
   pip install pylint
   ```

2. Run Pylint locally:
   ```bash
   pylint your_python_file.py
   ```

3. To run Pylint on the entire repository:
   ```bash
   pylint .
   ```

#### **Linting Jupyter Notebooks**
Before committing your code, ensure it passes all checks using nbQA for Notebooks code quality:
1. Install :
   ```bash
   pip install nbQA
   ```

2. To lint a Jupyter notebook:
   ```bash
   nbQA pylint your_notebook.ipynb
   ```

3. To run nbQA on the entire repository:
   ```bash
   nbQA pylint .
   ```

### 4. Cyclomatic Complexity Testing for Python Files and Jupyter Notebooks

#### **Maintainability Index for Python Files**
Before committing your code, ensure it passes all checks using Radon for measuring maintainability index:

1. Install Radon:
   ```bash
   pip install radon
   ```

2. Run Radon locally to measure measuring maintainability:
   ```bash
   radon mi your_python_file.py
   ```

2. To run Radon on the entire repository:
   ```bash
   radon mi .
   ```

#### **Maintainability Index for Jupyter Notebooks**
Before committing your code, ensure it passes all checks using Radon for measuring maintainability index:

1. Install Radon:
   ```bash
   pip install radon
   ```

2. Run Radon locally to measure measuring maintainability:
   ```bash
   radon mi --include-ipynb your_notebook.ipynb
   ```

2. To run Radon on the entire repository:
   ```bash
   radon mi --include-ipynb .
   ```
---

### 5. Pylint, Radon, and Black Checks

Before committing your code, ensure it passes all checks by running the appropriate commands for **Black**, **Radon**, and **Pylint** as part of the repository's Makefile.

#### **Running All Checks**

1. Use the Makefile's `check` command to validate all checks:
   ```bash
   make check
   ```

2. Fix any errors indicated by Black, Radon, or Pylint before committing your code.

---

### 6. Commit Messages
1. Write clear, descriptive commit messages.
2. Follow this format:
   ```
   feat: Add a new feature
   fix: Fix an issue
   docs: Update documentation
   style: Format code (e.g., Black formatting)
   refactor: Code refactoring
   test: Add or update tests
   chore: Maintenance work
   ```

---

### 7. Submitting Your Contribution
1. Push your branch to your forked repository:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Create a pull request to the main repository. Note you will need to fill out a pull request form.
3. Ensure your PR passes all checks, including Black formatting, Pylint linting, and Radon maintainability checks.

---

### 8. Additional Notes
- Contributions that fail any of the checks (Black, Pylint, or Radon) will not be accepted until they are resolved.
- For questions or further assistance, feel free to open an issue or contact the maintainers.

Thank you for contributing!
