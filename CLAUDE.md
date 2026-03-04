# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Model Basics** is an interactive HTML5 slideshow presentation (no build process required) that teaches 23 concepts about how modern AI models are built, trained, and used. The presentation flows through architecture, training, inference, and advanced capabilities.

## Architecture & File Organization

The project is a pure **client-side web application** with a simple, flat structure:

- **index.html** — Main entry point, defines DOM structure (nav, slides container, script tags)
- **model-basics.js** — Slideshow controller: renders slides from data, handles navigation (keyboard, buttons, touch), manages URL hash routing
- **cards-data.js** — All slide content as a JavaScript array (`cardsData`); each object represents one slide with title, description, bullets, paragraphs, callouts, and curated resources
- **model-basics.css** — All styling; uses CSS Grid for responsive layout, handles slide transitions with `transform: translateX()`
- **docs/** — Supplementary materials: `cheat-sheet.html` (rendered reference) and `cheat-sheet.yaml` (source data)

## Key Architecture Patterns

### Slide Data Structure
Each item in `cardsData` array follows this schema:
```javascript
{
    category: string,              // 'arch', 'train', 'infer', 'adv'
    badge: string,                 // Short label (e.g., "Architecture")
    title: string,                 // Slide title (numbered)
    description?: string,          // Optional short intro
    paragraphs?: string[],         // Main content paragraphs (supports HTML)
    bullets?: string[],            // Bullet points (supports HTML)
    callout?: {                    // Optional highlighted box
        type: 'insight' | 'analogy' | 'note',
        content: string            // HTML content
    },
    resources?: {                  // Curated external links
        type: 'video' | 'article' | 'tool' | 'interactive',
        title: string,
        meta?: string,             // Credit/duration
        url: string,
        icon?: string              // Custom icon (defaults by type)
    }[],
    image?: {                      // Optional slide image (opens in lightbox)
        url: string,
        caption: string,
        attribution: string
    }
}
```

### Navigation & State
- **Current slide state** lives in `currentSlide` (index, 0-based)
- **URL hash** (`#slide-N`) reflects position; hash changes trigger re-render
- Navigation works via: arrow keys, nav buttons, touch swipe, or direct hash link

### Rendering
- `renderSlides()` builds all slides once (no lazy loading)
- `showSlide(index)` translates the container left using CSS `transform` and updates hash
- Slides are 100vw wide, container scrolls horizontally

## Important Rules & Conventions

### Version Numbering
When `cards-data.js` changes (any content update), manually bump the version in `index.html`:
```html
<span class="version">vN</span>  <!-- increment N, e.g., v10 → v11 -->
```
This should be done in the same commit as the content change to keep the displayed version in sync with slide content.

### No Build Process
- This is a vanilla HTML/CSS/JS project with **zero dependencies**
- Open `index.html` directly in any modern browser
- All content is inline or in script tags; no bundlers or compilation needed

### Browser & Device Support
- Modern desktop browsers: Chrome, Firefox, Safari, Edge (latest)
- Mobile: iOS Safari, Chrome Mobile (all features including swipe)
- All navigation patterns work: keyboard, mouse, touch

## Common Development Tasks

### Adding or Editing Slides
1. Open `cards-data.js` and locate the `cardsData` array
2. Add a new object or edit existing one with appropriate fields
3. Update `index.html` version number (e.g., `v10` → `v11`)
4. Test by opening `index.html` in browser; use arrow keys or URL hash to navigate

### Testing Navigation
- **Keyboard:** Arrow keys (← →) to move between slides
- **URL hash:** Manually change `#slide-5` in address bar to jump to slide 5
- **Touch:** Swipe left/right on mobile or trackpad
- Progress bar and slide counter update automatically

### Resources
- Resource links are curated external URLs; they open in new tabs
- Resource types (video, article, tool, interactive) have built-in icon defaults
- Can override icons with custom `icon` field if needed

### Styling
- Single stylesheet (`model-basics.css`) defines all layout and theming
- Responsive design uses CSS Grid and media queries
- Slide transitions use `transform` (GPU-accelerated) and CSS `transition`
- Callout types have distinct colors; keep consistent with existing palette

## Git Workflow & Conventions

- **Main branch:** `main` — always stable, deployed state
- **Commit messages:** Reference specific changes (e.g., "Add RAG slide details (v11)" or "Fix typo in Attention slide")
- **Version bumps:** Include version change in same commit as content edits
- **Docs:** Keep supplementary materials in `docs/` directory; link in README if significant

### Pre-Commit Alignment Check
Before every commit and push, verify that `README.md` and `CLAUDE.md` are consistent with the current state of the codebase:
- Slide count in both files matches the actual number of entries in `cardsData`
- Topics list in `README.md` matches the actual slide titles and order in `cards-data.js`
- Version number mentioned anywhere matches `<span class="version">` in `index.html`
- File structure descriptions reflect actual files present
- Any new slide fields or schema changes are reflected in the `CLAUDE.md` data structure schema

## Testing

### Automated Tests
Run tests by opening `tests.html` in a browser. The test suite covers:
- **Data validation:** All slides have required fields, valid categories, callout types, and resource URLs
- **Slideshow logic:** Hash parsing, CONFIG properties, element caching

### Manual Browser Testing
Before committing changes involving slide content or navigation:
- [ ] All slides render without JS errors (check browser console)
- [ ] Navigation works with arrow keys
- [ ] URL hash updates when navigating; direct hash links work
- [ ] Progress bar reflects correct position
- [ ] Resource links open in new tabs
- [ ] Layout is responsive (test on narrow browser window or phone)