# FedCast Documentation

This directory contains the documentation website for FedCast, built with Jekyll and deployed via GitHub Pages.

## Local Development

To run the documentation site locally:

1. **Install Ruby and Bundler** (if not already installed):
   ```bash
   # macOS with Homebrew
   brew install ruby
   
   # Ubuntu/Debian
   sudo apt-get install ruby ruby-dev build-essential
   
   # Windows - use RubyInstaller
   ```

2. **Install dependencies**:
   ```bash
   cd docs
   bundle install
   ```

3. **Run the development server**:
   ```bash
   bundle exec jekyll serve
   ```

4. **View the site**: Open [http://localhost:4000](http://localhost:4000) in your browser

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by the GitHub Actions workflow in `.github/workflows/pages.yml`.

## Structure

- `_config.yml` - Jekyll configuration
- `index.html` - Homepage
- `docs/` - Documentation pages
  - `index.md` - Main documentation
  - `api/` - API reference
  - `examples/` - Usage examples
- `assets/` - Images, CSS, and other static files
- `_layouts/` - Jekyll layout templates
- `Gemfile` - Ruby dependencies

## Adding Content

### Adding a new documentation page:

1. Create a new `.md` file in the appropriate directory
2. Add front matter at the top:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   ---
   ```
3. Write your content in Markdown

### Adding a new example:

1. Create a new file in `docs/examples/`
2. Follow the existing example format
3. Include code blocks with syntax highlighting

### Updating the homepage:

Edit `index.html` to modify the homepage content and styling.

## Customization

- **Styling**: Modify `assets/css/custom.css`
- **Layout**: Update `_layouts/default.html`
- **Navigation**: Edit the `navigation` section in `_config.yml`
- **Configuration**: Modify `_config.yml` for site-wide settings

## Contributing

When contributing to the documentation:

1. Make your changes in the `docs/` directory
2. Test locally with `bundle exec jekyll serve`
3. Commit and push your changes
4. The site will be automatically deployed to GitHub Pages

For more information about Jekyll, see the [Jekyll documentation](https://jekyllrb.com/docs/).
