# Contributing to **RAGBee FW**

Thank you for your interest in contributing!  
This guide describes a minimal workflow so your Pull Request (PR) passes review smoothly.

> ## TL;DR
>
> ```bash
> poetry install --with dev
> git checkout -b feat/my-cool-thing
> # … hack on code …
> pre-commit run --all-files
> pytest                 # TODO ⇢ no test-suite yet, see **Tests** section
> git commit -m "feat: my cool thing"
> git push --set-upstream origin feat/my-cool-thing
> ```

---

## 1. Clone the repository

```bash
git clone https://github.com/<your-account>/ragbee_fw.git
cd ragbee_fw
```

---

## 2. Local setup

Requires **Python ≥ 3.10** and **Poetry ≥ 1.7**.

```bash
poetry install --with dev           # prod + dev dependencies
poetry run pre-commit install       # enable git hooks (Black, isort, Ruff)
```

---

## 3. Create a feature branch

```bash
git checkout -b feat/<short-topic>
```

Allowed prefixes: `feat/*`, `fix/*`, `docs/*`, `infra/*`.

---

## 4. Code style & static analysis

* **Git hooks** auto‑run Black + isort + Ruff.  
* Type checking:

```bash
poetry run mypy src/
```

---

## 5. Tests **(TODO)**

The project currently has no test-suite.  
Once the `tests/` folder appears, **every PR must include tests for new functionality**.

```bash
# future step
poetry run pytest -q
```

---

## 6. Commits

Follow **Conventional Commits**:

```
feat(splitter): add recursive text splitter
fix(retriever): handle empty index when no docs
docs(readme): add English quickstart
```

Add extra details after an empty line if needed.

---

## 7. Pull Request

1. **No force‑push** — first push like this:

   ```bash
   git push --set-upstream origin feat/<name>
   ```

2. PR description should include  
   - **What** and **why**;  
   - Link to an issue (if any);  
   - A checklist:

     ```markdown
     - [x] Code passes linters
     - [ ] Tests added   # (once tests/ exist)
     ```

3. **CI / GitHub Actions — TODO**  
   A future CI pipeline will automatically run  
   linters → mypy → pytest → `poetry build`.  
   Until then, reviewers will run checks locally.

---

## 8. Versioning & releases

We merge via **Squash & Merge** (linear history).  
After merging, a maintainer bumps the package version:

```bash
poetry version patch   # or minor / major
git tag v<new-version>
git push --tags
```

*(When CI is ready, GitHub Actions will publish the new version to PyPI — **TODO**).*

---

## 9. Code of Conduct

We follow the standards in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

**Thank you for helping RAGBee FW grow! 🚀**
