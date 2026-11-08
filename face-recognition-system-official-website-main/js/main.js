// ===== Smooth Scroll for Nav Links =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const href = this.getAttribute('href');
    if (href === '#') return;
    
    const target = document.querySelector(href);
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      
      // Close mobile menu if open
      const navMenu = document.getElementById('nav-menu');
      const hamburger = document.getElementById('hamburger');
      if (navMenu.classList.contains('active')) {
        navMenu.classList.remove('active');
        hamburger.classList.remove('active');
      }
    }
  });
});

// ===== Hamburger Menu Toggle =====
const hamburger = document.getElementById('hamburger');
const navMenu = document.getElementById('nav-menu');

if (hamburger) {
  hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
  });
}

// ===== Active Nav Link on Scroll =====
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-menu a');

window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(section => {
    const sectionTop = section.offsetTop - 100;
    if (window.scrollY >= sectionTop) {
      current = section.getAttribute('id');
    }
  });

  navLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === `#${current}`) {
      link.classList.add('active');
    }
  });
});

// ===== Tab Switching for Demo Section =====
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const tabId = btn.getAttribute('data-tab');
    
    // Remove active from all buttons
    tabBtns.forEach(b => b.classList.remove('active'));
    // Remove active from all contents
    tabContents.forEach(c => c.classList.remove('active'));
    
    // Add active to clicked button
    btn.classList.add('active');
    // Add active to matching content
    const targetContent = document.getElementById(`${tabId}-tab`);
    if (targetContent) {
      targetContent.classList.add('active');
    }
  });
});

// ===== Copy Code Function =====
function copyCode(btn) {
  const codeBox = btn.parentElement;
  const code = codeBox.querySelector('code').textContent;
  
  navigator.clipboard.writeText(code).then(() => {
    btn.classList.add('copied');
    btn.innerHTML = '<i class="fas fa-check"></i>';
    
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = '<i class="fas fa-copy"></i>';
    }, 2000);
  }).catch(() => {
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = code;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    
    btn.classList.add('copied');
    btn.innerHTML = '<i class="fas fa-check"></i>';
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = '<i class="fas fa-copy"></i>';
    }, 2000);
  });
}

// Make copyCode available globally
window.copyCode = copyCode;

// ===== Scroll to Top Button =====
const scrollTopBtn = document.getElementById('scroll-top');

window.addEventListener('scroll', () => {
  if (window.scrollY > 400) {
    scrollTopBtn.classList.add('show');
  } else {
    scrollTopBtn.classList.remove('show');
  }
});

if (scrollTopBtn) {
  scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// ===== Intersection Observer for Scroll Reveal =====
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, {
  threshold: 0.15,
  rootMargin: '0px 0px -50px 0px'
});

// Add reveal class to elements and observe
document.querySelectorAll(
  '.feature-card, .download-card, .doc-card, .info-card, .capability, .requirement, .system-requirements, .demo-showcase'
).forEach((el, i) => {
  el.classList.add('reveal');
  el.style.transitionDelay = `${(i % 4) * 0.1}s`;
  revealObserver.observe(el);
});

// ===== Parallax Floating Background Shapes =====
const shapes = document.querySelectorAll('.shape');
let mouseTimer = null;

window.addEventListener('mousemove', (e) => {
  if (mouseTimer) return;
  
  mouseTimer = setTimeout(() => {
    const x = (e.clientX / window.innerWidth - 0.5) * 2;
    const y = (e.clientY / window.innerHeight - 0.5) * 2;

    shapes.forEach((shape, i) => {
      const speed = (i + 1) * 5;
      shape.style.transform = `translate(${x * speed}px, ${y * speed}px)`;
    });
    
    mouseTimer = null;
  }, 16);
});

// ===== 3D Tilt Effect on Feature Cards =====
document.querySelectorAll('.feature-card').forEach(card => {
  card.addEventListener('mousemove', (e) => {
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const rotateX = (y - cy) / 20;
    const rotateY = (cx - x) / 20;

    card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
  });

  card.addEventListener('mouseleave', () => {
    card.style.transform = '';
  });
});

// ===== Animated Confidence & Liveness Value in Hero =====
const confidenceTag = document.querySelector('.confidence-tag');
if (confidenceTag) {
  const livenessBadges = [
    '🟢 99.4% • Liveness PASS',
    '🟢 98.9% • Real Human',
    '🟢 99.2% • Blink: 2 (EAR)',
    '🟢 99.7% • Yaw: +1.2°'
  ];
  let badgeIdx = 0;
  setInterval(() => {
    badgeIdx = (badgeIdx + 1) % livenessBadges.length;
    confidenceTag.textContent = livenessBadges[badgeIdx];
  }, 2500);
}

// ===== Random Wobble for Tech/Capability Cards =====
document.querySelectorAll('.capability, .info-card').forEach((card, i) => {
  const wobble = () => {
    const rotation = (Math.random() - 0.5) * 3;
    card.style.transform = `rotate(${rotation}deg)`;
    setTimeout(() => {
      card.style.transform = '';
    }, 1000);
  };

  setInterval(wobble, 4000 + i * 600);
});

// ===== Screenshot Error Handling =====
const screenshot = document.querySelector('.screenshot-img');
if (screenshot) {
  screenshot.addEventListener('error', () => {
    screenshot.style.display = 'none';
    screenshot.parentElement.classList.add('img-error');
  });
}

// ===== Console Greeting =====
console.log(`
╔══════════════════════════════════════════════════════════╗
║  🤖 Face Recognition System v2.1.0                       ║
║  2026 Enterprise Biometric AI Platform                   ║
║  FastAPI • Anti-Spoofing • RTSP • Apple Face ID Wizard   ║
║  74 / 74 Unit Tests Passing (2.08s)                      ║
║  Created with 💜 by Nilay Naha                           ║
╚══════════════════════════════════════════════════════════╝
`);

// ===== Animated Stats Counter =====
const animateCounter = (element, target, suffix = '', duration = 1500) => {
  const isFloat = target % 1 !== 0;
  let start = 0;
  const increment = target / (duration / 16);
  
  const update = () => {
    start += increment;
    if (start < target) {
      element.textContent = (isFloat ? start.toFixed(1) : Math.floor(start)) + suffix;
      requestAnimationFrame(update);
    } else {
      element.textContent = (isFloat ? target.toFixed(1) : target) + suffix;
    }
  };
  update();
};

// Animate stats when visible
const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const statNumbers = entry.target.querySelectorAll('.stat-number');
      statNumbers.forEach(stat => {
        const text = stat.textContent;
        if (text.includes('99') || text.includes('95')) {
          animateCounter(stat, 99.2, '%+');
        } else if (text.includes('60') || text.includes('30')) {
          animateCounter(stat, 60, '+ FPS');
        } else if (text.includes('74')) {
          animateCounter(stat, 74, ' / 74');
        }
      });
      statsObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
  statsObserver.observe(heroStats);
}

// ===== Navbar Shadow Enhancement on Scroll =====
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  if (window.scrollY > 20) {
    navbar.style.boxShadow = '0 6px 0 rgba(45, 27, 78, 0.2)';
  } else {
    navbar.style.boxShadow = '0 4px 0 rgba(45, 27, 78, 0.15)';
  }
});