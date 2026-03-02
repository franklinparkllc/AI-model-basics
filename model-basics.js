// Model Basics - Simple Slideshow

// Configuration
const CONFIG = {
    TOUCH_THRESHOLD: 50,           // pixels required to trigger slide change
    HASH_PREFIX: '#slide-',
    RESOURCE_TYPES: {
        video: '▶',
        article: '◇',
        tool: '◆',
        interactive: '◈'
    }
};

// State
let currentSlide = 0;
let slides = [];
let totalSlides = 0;

// DOM elements (cached for performance)
let elementCache = {};

function getElement(id) {
    if (!elementCache[id]) {
        elementCache[id] = document.getElementById(id);
    }
    return elementCache[id];
}

// Parse slide number from hash safely
function parseHashSlide(hash) {
    if (!hash || !hash.startsWith(CONFIG.HASH_PREFIX)) {
        return null;
    }
    const num = parseInt(hash.replace(CONFIG.HASH_PREFIX, ''), 10);
    return isNaN(num) ? null : num - 1; // convert to 0-based index
}

// Initialize
function init() {
    // Ensure cardsData is available
    if (typeof cardsData === 'undefined' || !cardsData || cardsData.length === 0) {
        console.error('cardsData is not available');
        return;
    }

    slides = cardsData;
    totalSlides = slides.length;

    if (totalSlides === 0) {
        console.error('No slides to render');
        return;
    }

    renderSlides();
    updateUI();
    setupEvents();

    // Handle URL hash
    const hash = window.location.hash;
    const hashSlide = parseHashSlide(hash);
    if (hashSlide !== null && hashSlide >= 0 && hashSlide < totalSlides) {
        currentSlide = hashSlide;
        showSlide(currentSlide);
    } else {
        // Ensure first slide is visible
        showSlide(0);
    }
}

// Render all slides
function renderSlides() {
    const container = getElement('slidesContainer');
    if (!container) {
        console.error('slidesContainer element not found');
        return;
    }

    // Clear container
    container.innerHTML = '';

    slides.forEach((card, index) => {
        const slide = document.createElement('div');
        slide.className = 'slide';
        slide.id = `slide-${index + 1}`;
        slide.setAttribute('data-category', card.category);

        let html = `
            <div class="slide-content">
                ${card.image ? `<button class="slide-img-btn" data-slide-index="${index}" aria-label="View illustration: ${card.image.caption}" title="View illustration">&#128247;</button>` : ''}
                <div class="slide-badge">${card.badge}</div>
                <h1 class="slide-title">${card.title}</h1>
                ${card.description ? `<p class="slide-description">${card.description}</p>` : ''}
                <div class="slide-body">
        `;

        // Paragraphs
        if (card.paragraphs) {
            card.paragraphs.forEach(p => {
                html += `<p>${p}</p>`;
            });
        }

        // Bullets
        if (card.bullets) {
            html += '<ul>';
            card.bullets.forEach(bullet => {
                html += `<li>${bullet}</li>`;
            });
            html += '</ul>';
        }

        // Callout
        if (card.callout) {
            html += `<div class="callout callout-${card.callout.type}">${card.callout.content}</div>`;
        }

        // Resources
        if (card.resources && card.resources.length > 0) {
            html += '<div class="resources">';
            html += '<div class="resources-header"><span class="resources-label">Resources</span></div>';
            html += '<div class="resources-list">';
            card.resources.forEach(res => {
                const type = res.type || 'article';
                const icon = res.icon || CONFIG.RESOURCE_TYPES[type] || CONFIG.RESOURCE_TYPES.article;
                html += `
                    <a href="${res.url}" target="_blank" rel="noopener" class="resource-link resource-link--${type}">
                        <span class="resource-icon" aria-hidden="true">${icon}</span>
                        <div class="resource-info">
                            <span class="resource-title">${res.title}</span>
                            ${res.meta ? `<span class="resource-meta">${res.meta}</span>` : ''}
                        </div>
                    </a>
                `;
            });
            html += '</div></div>';
        }

        html += '</div></div>';
        slide.innerHTML = html;
        container.appendChild(slide);
    });
}

// Show specific slide
function showSlide(index) {
    if (index < 0 || index >= totalSlides) {
        console.warn('Invalid slide index:', index);
        return;
    }

    currentSlide = index;
    const container = getElement('slidesContainer');
    if (!container) {
        console.error('slidesContainer element not found');
        return;
    }

    container.style.transform = `translateX(-${index * 100}vw)`;
    window.location.hash = `${CONFIG.HASH_PREFIX}${index + 1}`;
    updateUI();
}

// Navigation
function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        showSlide(currentSlide + 1);
    }
}

function prevSlide() {
    if (currentSlide > 0) {
        showSlide(currentSlide - 1);
    }
}

// Update UI
function updateUI() {
    const progress = ((currentSlide + 1) / totalSlides) * 100;
    getElement('progressFill').style.width = `${progress}%`;
    getElement('progressText').textContent = `${currentSlide + 1} / ${totalSlides}`;

    getElement('prevBtn').disabled = currentSlide === 0;
    getElement('nextBtn').disabled = currentSlide === totalSlides - 1;
}

// --- Image Modal ---

function openImageModal(image) {
    const modal = document.getElementById('imgModal');
    document.getElementById('imgModalImg').src = image.url;
    document.getElementById('imgModalImg').alt = image.caption || '';
    document.getElementById('imgModalCaption').textContent = image.caption || '';
    document.getElementById('imgModalAttribution').textContent = image.attribution || '';
    modal.hidden = false;
}

function closeImageModal() {
    const modal = document.getElementById('imgModal');
    if (!modal.hidden) {
        modal.hidden = true;
        document.getElementById('imgModalImg').src = '';
    }
}

// Setup event listeners
function setupEvents() {
    getElement('homeBtn').addEventListener('click', () => showSlide(0));
    getElement('prevBtn').addEventListener('click', prevSlide);
    getElement('nextBtn').addEventListener('click', nextSlide);

    // Keyboard
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') prevSlide();
        if (e.key === 'ArrowRight') nextSlide();
        if (e.key === 'Escape') closeImageModal();
    });

    // Touch swipe
    let startX = 0;
    let endX = 0;

    const container = getElement('slidesContainer');

    container.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
    });

    container.addEventListener('touchend', (e) => {
        endX = e.changedTouches[0].clientX;
        const diff = startX - endX;
        if (Math.abs(diff) > CONFIG.TOUCH_THRESHOLD) {
            if (diff > 0) nextSlide();
            else prevSlide();
        }
    });

    // Image lightbox — delegated from slides container
    container.addEventListener('click', (e) => {
        const btn = e.target.closest('.slide-img-btn');
        if (!btn) return;
        const slideIndex = parseInt(btn.getAttribute('data-slide-index'), 10);
        const card = slides[slideIndex];
        if (card && card.image) {
            openImageModal(card.image);
        }
    });

    // Modal close — X button
    document.getElementById('imgModalClose').addEventListener('click', closeImageModal);

    // Modal close — backdrop click
    document.querySelector('.img-modal-backdrop').addEventListener('click', closeImageModal);

    // Hash change (browser back/forward)
    window.addEventListener('hashchange', () => {
        const hashSlide = parseHashSlide(window.location.hash);
        if (hashSlide !== null && hashSlide >= 0 && hashSlide < totalSlides && hashSlide !== currentSlide) {
            showSlide(hashSlide);
        }
    });
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
