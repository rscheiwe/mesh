# Mesh Documentation

This directory contains the documentation for Mesh, built with Jekyll and the Just the Docs theme.

## Local Development

### Prerequisites

- Ruby 2.7 or higher
- Bundler

### Setup

```bash
# Navigate to docs directory
cd docs

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# View at http://localhost:4000/mesh
```

### Building for Production

```bash
bundle exec jekyll build
# Output in _site/
```

## Deploying to GitHub Pages

### Option 1: Automatic (Recommended)

1. Push to GitHub:
   ```bash
   git add docs/
   git commit -m "Add documentation"
   git push
   ```

2. Enable GitHub Pages in repository settings:
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` or `master`
   - Folder: `/docs`
   - Save

3. Site will be available at: `https://rscheiwe.github.io/mesh`

### Option 2: GitHub Actions

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.1'

      - name: Install dependencies
        run: |
          cd docs
          bundle install

      - name: Build site
        run: |
          cd docs
          bundle exec jekyll build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_site
```

## Documentation Structure

```
docs/
├── _config.yml           # Jekyll configuration
├── Gemfile               # Ruby dependencies
├── _sass/                # Custom styles
│   └── custom/
│       └── custom.scss
├── index.md              # Home page
├── getting-started.md    # Installation guide
├── quick-start.md        # Quick examples
├── concepts/             # Core concepts
│   ├── graphs.md
│   ├── nodes.md
│   ├── execution.md
│   └── events.md
├── guides/               # How-to guides
│   ├── streaming.md
│   ├── state-management.md
│   ├── variables.md
│   └── conditional-branching.md
├── integrations/         # Integrations
│   ├── vel.md
│   ├── openai-agents.md
│   └── event-translation.md
├── api-reference/        # API docs
│   └── ...
├── examples.md           # Examples
└── troubleshooting.md    # Common issues
```

## Contributing to Docs

### Adding a New Page

1. Create a markdown file
2. Add YAML front matter:
   ```yaml
   ---
   layout: default
   title: Page Title
   nav_order: 5
   ---
   ```

3. For child pages:
   ```yaml
   ---
   layout: default
   title: Child Page
   parent: Parent Page
   nav_order: 1
   ---
   ```

### Navigation Order

Pages are ordered by `nav_order` in front matter. Lower numbers appear first.

### Styling

Custom styles go in `_sass/custom/custom.scss`.

## Links

- [Live Documentation](https://rscheiwe.github.io/mesh)
- [GitHub Repository](https://github.com/rscheiwe/mesh)
- [Just the Docs Theme](https://just-the-docs.github.io/just-the-docs/)
