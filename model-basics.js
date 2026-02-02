// Model Basics - Simple Slideshow

let currentSlide = 0;
let slides = [];
let totalSlides = 0;

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
    if (hash) {
        const index = parseInt(hash.replace('#slide-', '')) - 1;
        if (index >= 0 && index < totalSlides) {
            currentSlide = index;
            showSlide(currentSlide);
        }
    } else {
        // Ensure first slide is visible
        showSlide(0);
    }
}

// Render all slides
function renderSlides() {
    const container = document.getElementById('slidesContainer');
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
            const typeIcons = { video: '▶', article: '◇', tool: '◆', interactive: '◈' };
            html += '<div class="resources">';
            html += '<div class="resources-header"><span class="resources-label">Resources</span></div>';
            html += '<div class="resources-list">';
            card.resources.forEach(res => {
                const type = res.type || 'article';
                const icon = res.icon || typeIcons[type] || '◇';
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
    const container = document.getElementById('slidesContainer');
    if (!container) {
        console.error('slidesContainer element not found');
        return;
    }
    
    container.style.transform = `translateX(-${index * 100}vw)`;
    window.location.hash = `slide-${index + 1}`;
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
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressText').textContent = `${currentSlide + 1} / ${totalSlides}`;
    
    document.getElementById('prevBtn').disabled = currentSlide === 0;
    document.getElementById('nextBtn').disabled = currentSlide === totalSlides - 1;
}

// Setup event listeners
function setupEvents() {
    document.getElementById('homeBtn').addEventListener('click', () => showSlide(0));
    document.getElementById('prevBtn').addEventListener('click', prevSlide);
    document.getElementById('nextBtn').addEventListener('click', nextSlide);
    
    // Keyboard
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') prevSlide();
        if (e.key === 'ArrowRight') nextSlide();
    });
    
    // Touch swipe
    let startX = 0;
    let endX = 0;
    
    const container = document.getElementById('slidesContainer');
    
    container.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
    });
    
    container.addEventListener('touchend', (e) => {
        endX = e.changedTouches[0].clientX;
        const diff = startX - endX;
        if (Math.abs(diff) > 50) {
            if (diff > 0) nextSlide();
            else prevSlide();
        }
    });
    
    // Hash change (browser back/forward)
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash;
        if (hash) {
            const index = parseInt(hash.replace('#slide-', '')) - 1;
            if (index >= 0 && index < totalSlides && index !== currentSlide) {
                showSlide(index);
            }
        }
    });
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
