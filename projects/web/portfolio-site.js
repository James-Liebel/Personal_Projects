(() => {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const progressBar = document.getElementById("topProgress");
  const siteNav = document.getElementById("siteNav");
  const navLinks = [...document.querySelectorAll('.nav-link[href^="#"]')];
  const sections = [...document.querySelectorAll("main section[id]")];
  const navIndicator = document.getElementById("navIndicator");
  const modeButtons = [...document.querySelectorAll(".switch button")];
  const journeyButtons = [...document.querySelectorAll(".rail button")];
  const journeyPanels = [...document.querySelectorAll(".jpanel")];
  const projectButtons = [...document.querySelectorAll(".project-rail button")];
  const resumeButtons = [...document.querySelectorAll(".resume-tabs button")];
  const resumePanels = [...document.querySelectorAll(".rpanel")];
  const mobileNav = document.getElementById("mobileNav");
  const navToggle = document.getElementById("navToggle");
  const navClose = document.getElementById("navClose");
  const hero = document.getElementById("hero");
  const heroTitle = document.getElementById("heroTitle");
  const heroWord = document.getElementById("heroWord");
  const heroPills = document.getElementById("heroPills");
  const heroPillsMarkup = heroPills ? heroPills.innerHTML : "";
  const signalTags = document.getElementById("signalTags");
  const journeyIndicator = document.getElementById("journeyIndicator");
  const resumeIndicator = document.getElementById("resumeIndicator");
  const modeIndicator = document.getElementById("modeIndicator");
  const projectDisplay = document.querySelector(".project-display");
  const consoleStack = document.querySelector(".stack");
  const pdfButtons = [document.getElementById("pdfBtn"), document.getElementById("pdfBtnMobile")].filter(Boolean);

  const modeData = {
    builder: {
      angle: 300,
      word: "Builder",
      label: "role view",
      title: "Best evidence for software and product work",
      badge: "Applications + ML projects",
      body: "This view highlights end-to-end work: data preparation, model implementation, workflow logic, interface design, and usable delivery.",
      s1: ["Core tools", "Python / SQL / FastAPI", "Main stack for analytics, prototypes, and lightweight application delivery."],
      s2: ["Shipped examples", "CustomStrat + JimAI", "Live site work and AI-assisted workflow prototypes."],
      s3: ["What stands out", "Readable interfaces", "Projects are shown as sites, maps, dashboards, or tools instead of flat notebooks."],
      foot: "Best fit for software, ML-adjacent, and product-minded internships.",
      tags: ["FastAPI", "React", "Next.js", "D3.js"]
    },
    analyst: {
      angle: 210,
      word: "Analyst",
      label: "role view",
      title: "Best evidence for data science and analytics roles",
      badge: "Modeling + evaluation",
      body: "This view highlights supervised learning, NLP, EDA, feature work, and recruiter-readable evaluation.",
      s1: ["Model work", "Fraud + sentiment", "Classification and text analysis projects show applied modeling experience."],
      s2: ["Evaluation", "Recall / benchmarks / tradeoffs", "Results are framed with metrics, comparisons, and tradeoffs."],
      s3: ["Communication", "Charts + writeups", "Visuals and plain-language explanations support the technical work."],
      foot: "Best fit for data science, analytics, and research-oriented internships.",
      tags: ["Scikit-learn", "XGBoost", "EDA", "TF-IDF"]
    },
    operator: {
      angle: 120,
      word: "Delivery",
      label: "role view",
      title: "Best evidence for execution and follow-through",
      badge: "Project structure + shipped outputs",
      body: "This view highlights finishing work well: organizing projects, documenting decisions, and shipping interfaces another person can navigate.",
      s1: ["Workflow habits", "Git + iteration", "Versioning, cleanup, and repeated refinement matter across the portfolio."],
      s2: ["Output style", "Sites + dashboards", "Work is packaged in a form that is easier to review than raw notebooks alone."],
      s3: ["Team signal", "Readable handoff", "The projects show an effort to make technical work understandable to other people."],
      foot: "Best fit for roles where follow-through and usable outputs matter as much as experimentation.",
      tags: ["Documentation", "Automation", "GitHub Pages", "Prompting"]
    }
  };

  const ring = document.getElementById("signalRing");
  const ringScore = document.getElementById("ringScore");
  const ringLabel = document.getElementById("ringLabel");
  const ringFoot = document.getElementById("ringFoot");
  const signalTitle = document.getElementById("signalTitle");
  const signalBadge = document.getElementById("signalBadge");
  const signalBody = document.getElementById("signalBody");
  const s1l = document.getElementById("s1l");
  const s1v = document.getElementById("s1v");
  const s1n = document.getElementById("s1n");
  const s2l = document.getElementById("s2l");
  const s2v = document.getElementById("s2v");
  const s2n = document.getElementById("s2n");
  const s3l = document.getElementById("s3l");
  const s3v = document.getElementById("s3v");
  const s3n = document.getElementById("s3n");

  const plabel = document.getElementById("plabel");
  const ptitle = document.getElementById("ptitle");
  const pcopy = document.getElementById("pcopy");
  const pal = document.getElementById("pal");
  const pav = document.getElementById("pav");
  const pbl = document.getElementById("pbl");
  const pbv = document.getElementById("pbv");
  const pcl = document.getElementById("pcl");
  const pcv = document.getElementById("pcv");
  const plink = document.getElementById("plink");
  const po = document.getElementById("poutline");
  const ptags = document.getElementById("ptags");

  function updateProgress() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progressBar.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
  }

  function setStagger(selector, step, revealType = "up") {
    document.querySelectorAll(selector).forEach((el, index) => {
      el.setAttribute("data-reveal", revealType);
      el.style.setProperty("--delay", `${index * step}ms`);
    });
  }

  function applyRevealHooks() {
    document.querySelectorAll(".reveal").forEach(el => el.setAttribute("data-reveal", "up"));
    document.querySelectorAll(".reveal-item").forEach(el => {
      if (!el.dataset.reveal) el.dataset.reveal = el.dataset.enter || "up";
    });
    setStagger(".cap", 80, "up");
    setStagger(".viz", 80, "up");
    setStagger(".project-rail button", 80, "left");
    setStagger(".resume-tabs button", 80, "left");
    setStagger(".rail button", 80, "left");
    setStagger(".guide-step", 80, "up");
    setStagger(".pmetric", 80, "scale");
    setStagger(".mini", 80, "scale");
    document.querySelectorAll(".head h2,.head p:last-child,.site-footer").forEach(el => {
      el.setAttribute("data-reveal", "up");
    });
    document.querySelectorAll(".metric strong,.mini strong,.pmetric strong").forEach(el => {
      el.setAttribute("data-reveal", "scale");
    });
  }

  applyRevealHooks();

  const revealTargets = [...document.querySelectorAll("[data-reveal]")];
  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("visible");
      revealObserver.unobserve(entry.target);
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });

  revealTargets.forEach(el => revealObserver.observe(el));

  function moveIndicator(indicator, button, options = {}) {
    if (!indicator || !button) return;
    const container = button.parentElement;
    if (!container) return;
    const containerRect = container.getBoundingClientRect();
    const buttonRect = button.getBoundingClientRect();
    const horizontal = options.horizontal ?? (window.innerWidth <= 720 && container.classList.contains("resume-tabs"));
    const x = buttonRect.left - containerRect.left;
    const y = buttonRect.top - containerRect.top;
    if (horizontal) {
      indicator.style.width = `${buttonRect.width}px`;
      indicator.style.height = `${Math.max(buttonRect.height - 8, 0)}px`;
      indicator.style.transform = `translate(${x}px, ${y + 4}px)`;
    } else if (container.classList.contains("switch")) {
      indicator.style.width = `${buttonRect.width}px`;
      indicator.style.height = `${buttonRect.height}px`;
      indicator.style.transform = `translate(${x}px, ${y}px)`;
    } else {
      indicator.style.width = "4px";
      indicator.style.height = `${buttonRect.height}px`;
      indicator.style.transform = `translate(${Math.max(x - 10, 0)}px, ${y}px)`;
    }
    indicator.classList.add("ready");
  }

  function updateNavIndicator(activeLink = document.querySelector(".nav-link.active")) {
    if (!navIndicator || !activeLink || window.innerWidth <= 760) return;
    const navRect = activeLink.parentElement.getBoundingClientRect();
    const linkRect = activeLink.getBoundingClientRect();
    navIndicator.style.width = `${linkRect.width}px`;
    navIndicator.style.transform = `translate(${linkRect.left - navRect.left}px, 2px)`;
    navIndicator.classList.add("ready");
  }

  function markActiveNav(id) {
    let activeLink = null;
    navLinks.forEach(link => {
      const isActive = link.getAttribute("href") === `#${id}`;
      link.classList.toggle("active", isActive);
      if (isActive) activeLink = link;
    });
    updateNavIndicator(activeLink);
  }

  const navObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) markActiveNav(entry.target.id);
    });
  }, { threshold: 0.48 });

  sections.forEach(section => navObserver.observe(section));

  function closeMobileNav() {
    if (!mobileNav || mobileNav.hidden) return;
    mobileNav.classList.remove("open");
    document.body.style.overflow = "";
    if (navToggle) {
      navToggle.classList.remove("active");
      navToggle.setAttribute("aria-expanded", "false");
    }
    window.setTimeout(() => {
      if (!mobileNav.classList.contains("open")) mobileNav.hidden = true;
    }, 360);
  }

  function openMobileNav() {
    if (!mobileNav) return;
    mobileNav.hidden = false;
    requestAnimationFrame(() => mobileNav.classList.add("open"));
    document.body.style.overflow = "hidden";
    if (navToggle) {
      navToggle.classList.add("active");
      navToggle.setAttribute("aria-expanded", "true");
    }
  }

  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", event => {
      const target = document.querySelector(anchor.getAttribute("href"));
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
      closeMobileNav();
    });
  });

  if (navToggle) navToggle.addEventListener("click", openMobileNav);
  if (navClose) navClose.addEventListener("click", closeMobileNav);
  if (mobileNav) {
    mobileNav.addEventListener("click", event => {
      if (event.target === mobileNav) closeMobileNav();
    });
  }
  document.querySelectorAll(".mobile-nav-links a").forEach(link => link.addEventListener("click", closeMobileNav));

  const heroWords = ["data science", "machine learning", "visualization", "analytics"];
  let heroWordIndex = 0;

  function swapWord(el, nextValue) {
    if (!el) return;
    el.classList.add("word-swap");
    window.setTimeout(() => {
      el.textContent = nextValue;
      el.classList.remove("word-swap");
    }, 180);
  }

  function cycleHeroWords() {
    heroWordIndex = (heroWordIndex + 1) % heroWords.length;
    swapWord(heroWord, heroWords[heroWordIndex]);
  }

  if (!reduced) window.setInterval(cycleHeroWords, 2800);

  if (!reduced && hero && heroTitle) {
    hero.addEventListener("pointermove", event => {
      const rect = hero.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width - 0.5) * 8;
      const y = ((event.clientY - rect.top) / rect.height - 0.5) * 8;
      heroTitle.style.setProperty("--hero-mx", `${x.toFixed(2)}px`);
      heroTitle.style.setProperty("--hero-my", `${y.toFixed(2)}px`);
    });
    hero.addEventListener("pointerleave", () => {
      heroTitle.style.setProperty("--hero-mx", "0px");
      heroTitle.style.setProperty("--hero-my", "0px");
    });
  }

  function updateHeroParallax() {
    if (!hero || reduced) return;
    const rect = hero.getBoundingClientRect();
    hero.style.setProperty("--hero-shift", `${Math.max(Math.min(rect.top * -0.08, 16), -16)}px`);
  }

  function setupHeroMarquee() {
    if (!heroPills) return;
    if (window.innerWidth > 760 || reduced) {
      if (heroPills.dataset.cloned === "true") {
        heroPills.innerHTML = heroPillsMarkup;
        heroPills.dataset.cloned = "false";
      }
      heroPills.classList.remove("is-marquee");
      return;
    }
    if (heroPills.dataset.cloned === "true") {
      heroPills.classList.add("is-marquee");
      return;
    }
    heroPills.innerHTML = `${heroPills.innerHTML}${heroPills.innerHTML}`;
    heroPills.dataset.cloned = "true";
    heroPills.classList.add("is-marquee");
  }

  function animateTags(container) {
    if (!container) return;
    [...container.querySelectorAll(".tag")].forEach((tag, index) => {
      tag.classList.add("is-entering");
      tag.style.transitionDelay = `${index * 80}ms`;
      requestAnimationFrame(() => tag.classList.add("visible"));
    });
  }

  function swapPanel(panel, updateFn) {
    if (!panel) {
      updateFn();
      return;
    }
    panel.classList.add("is-switching");
    window.setTimeout(() => {
      updateFn();
      panel.classList.remove("is-switching");
    }, reduced ? 0 : 180);
  }

  function setActivePanel(buttons, panels, activeButton, panelKey, panelAttr) {
    buttons.forEach(button => {
      const isActive = button === activeButton;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    panels.forEach(panel => panel.classList.toggle("active", panel.dataset[panelAttr] === panelKey));
  }

  function updateStaticIndicators() {
    moveIndicator(modeIndicator, document.querySelector(".switch button.active"));
    moveIndicator(journeyIndicator, document.querySelector(".rail button.active"));
    moveIndicator(resumeIndicator, document.querySelector(".resume-tabs button.active"));
    updateNavIndicator();
  }

  function renderMode(key) {
    const next = modeData[key];
    if (!next) return;
    const activeButton = modeButtons.find(button => button.dataset.mode === key);
    modeButtons.forEach(button => button.classList.toggle("active", button === activeButton));
    moveIndicator(modeIndicator, activeButton);
    swapPanel(consoleStack, () => {
      ring.style.setProperty("--angle", `${next.angle}deg`);
      ringScore.textContent = next.word;
      ringLabel.textContent = next.label;
      ringFoot.textContent = next.foot;
      signalTitle.textContent = next.title;
      signalBadge.textContent = next.badge;
      signalBody.textContent = next.body;
      [[s1l, s1v, s1n, next.s1], [s2l, s2v, s2n, next.s2], [s3l, s3v, s3n, next.s3]].forEach(([a, b, c, values]) => {
        a.textContent = values[0];
        b.textContent = values[1];
        c.textContent = values[2];
      });
      signalTags.innerHTML = next.tags.map(tag => `<span class="tag">${tag}</span>`).join("");
      animateTags(signalTags);
    });
  }

  modeButtons.forEach(button => button.addEventListener("click", () => renderMode(button.dataset.mode)));
  renderMode("builder");

  journeyButtons.forEach(button => {
    button.addEventListener("click", () => {
      setActivePanel(journeyButtons, journeyPanels, button, button.dataset.journey, "panel");
      moveIndicator(journeyIndicator, button);
    });
  });

  function renderProject(button) {
    if (!button) return;
    projectButtons.forEach(item => item.classList.toggle("active", item === button));
    swapPanel(projectDisplay, () => {
      plabel.textContent = button.dataset.label || "";
      ptitle.textContent = button.dataset.title || "";
      pcopy.textContent = button.dataset.copy || "";
      pal.textContent = button.dataset.aLabel || "";
      pav.textContent = button.dataset.aValue || "";
      pbl.textContent = button.dataset.bLabel || "";
      pbv.textContent = button.dataset.bValue || "";
      pcl.textContent = button.dataset.cLabel || "";
      pcv.textContent = button.dataset.cValue || "";
      plink.href = button.dataset.link || "#";
      plink.target = (button.dataset.link || "").startsWith("http") ? "_blank" : "_self";
      plink.rel = plink.target === "_blank" ? "noopener noreferrer" : "";
      po.innerHTML = (button.dataset.outline || "").split("|").filter(Boolean).map(line => `<li>${line}</li>`).join("");
      ptags.innerHTML = (button.dataset.stack || "").split(",").map(item => item.trim()).filter(Boolean).map(item => `<span class="tag">${item}</span>`).join("");
      animateTags(ptags);
    });
  }

  projectButtons.forEach(button => button.addEventListener("click", () => renderProject(button)));
  renderProject(projectButtons[0]);

  resumeButtons.forEach(button => {
    button.addEventListener("click", () => {
      setActivePanel(resumeButtons, resumePanels, button, button.dataset.resume, "rpanel");
      moveIndicator(resumeIndicator, button);
      if (button.dataset.resume === "summary") {
        animateTags(document.querySelector('.rpanel[data-rpanel="summary"] .resume-pills'));
      }
    });
  });

  animateTags(document.querySelector('.rpanel[data-rpanel="summary"] .resume-pills'));

  const countObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const target = Number(el.dataset.value || "0");
      const suffix = el.dataset.suffix || "";
      const duration = 1200;
      const start = performance.now();
      el.classList.add("counting");
      function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = Math.round(target * eased);
        el.textContent = `${value.toLocaleString()}${suffix}`;
        if (progress < 1) {
          requestAnimationFrame(step);
        } else {
          el.classList.remove("counting");
        }
      }
      requestAnimationFrame(step);
      countObserver.unobserve(el);
    });
  }, { threshold: 0.55 });

  document.querySelectorAll(".count-up").forEach(el => countObserver.observe(el));

  function setupVizFallbacks() {
    document.querySelectorAll(".viz-media").forEach(media => {
      const frame = media.querySelector("iframe");
      const fallback = media.querySelector(".viz-fallback");
      if (!frame || !fallback) return;
      let settled = false;
      const fail = () => {
        if (settled) return;
        settled = true;
        frame.hidden = true;
        fallback.hidden = false;
      };
      const pass = () => {
        if (settled) return;
        settled = true;
        fallback.hidden = true;
      };
      frame.addEventListener("load", () => {
        try {
          const doc = frame.contentDocument;
          const bodyText = doc && doc.body ? doc.body.innerText.trim().toLowerCase() : "";
          if (bodyText.includes("cannot find") || bodyText.includes("not found")) {
            fail();
            return;
          }
        } catch (error) {
          /* iframe preview fallback stays resilient if browser policies change */
        }
        pass();
      }, { once: true });
      frame.addEventListener("error", fail, { once: true });
      window.setTimeout(() => {
        if (!settled) fail();
      }, 3500);
    });
  }

  setupVizFallbacks();

  if (!reduced) {
    document.querySelectorAll(".panel,.card,.cap,.project-rail button,.viz").forEach(el => {
      el.addEventListener("pointermove", event => {
        const rect = el.getBoundingClientRect();
        el.style.setProperty("--cx", `${((event.clientX - rect.left) / rect.width) * 100}%`);
        el.style.setProperty("--cy", `${((event.clientY - rect.top) / rect.height) * 100}%`);
      });
    });
    window.addEventListener("pointermove", event => {
      document.body.style.setProperty("--px", `${event.clientX}px`);
      document.body.style.setProperty("--py", `${event.clientY}px`);
    }, { passive: true });
  }

  function fitResumeToOnePage() {
    const resume = document.getElementById("resume");
    const page = document.querySelector("#resume .resume-page");
    if (!resume || !page) return;
    const previous = {
      display: resume.style.display,
      position: resume.style.position,
      visibility: resume.style.visibility,
      left: resume.style.left,
      top: resume.style.top,
      width: resume.style.width
    };
    resume.style.display = "block";
    resume.style.position = "absolute";
    resume.style.visibility = "hidden";
    resume.style.left = "-10000px";
    resume.style.top = "0";
    const pageHeightIn = 11;
    const pageWidthIn = 8.5;
    const marginIn = 1.5 / 2.54;
    const printableHeightPx = (pageHeightIn - marginIn * 2) * 96;
    const printableWidthIn = pageWidthIn - marginIn * 2;
    resume.style.width = `${printableWidthIn}in`;
    page.style.transform = "";
    page.style.transformOrigin = "top center";
    page.style.marginBottom = "";
    const height = page.scrollHeight;
    let scale = 1;
    if (height > printableHeightPx) scale = Math.max(.78, printableHeightPx / height);
    page.style.transform = `scale(${scale.toFixed(3)})`;
    page.style.marginBottom = `-${(1 - scale) * height}px`;
    resume.style.display = previous.display;
    resume.style.position = previous.position;
    resume.style.visibility = previous.visibility;
    resume.style.left = previous.left;
    resume.style.top = previous.top;
    resume.style.width = previous.width;
  }

  function clearResumeScale() {
    const page = document.querySelector("#resume .resume-page");
    if (!page) return;
    page.style.transform = "";
    page.style.transformOrigin = "";
    page.style.marginBottom = "";
  }

  let printing = false;
  let printTimer = null;

  function restoreResume() {
    if (!printing) return;
    if (printTimer) {
      clearTimeout(printTimer);
      printTimer = null;
    }
    clearResumeScale();
    printing = false;
  }

  function printResume() {
    if (printing) return;
    printing = true;
    fitResumeToOnePage();
    window.addEventListener("afterprint", restoreResume, { once: true });
    window.addEventListener("focus", restoreResume, { once: true });
    printTimer = window.setTimeout(restoreResume, 4000);
    requestAnimationFrame(() => requestAnimationFrame(() => window.print()));
  }

  pdfButtons.forEach(button => button.addEventListener("click", printResume));
  window.printResume = printResume;

  function onScroll() {
    updateProgress();
    updateHeroParallax();
    if (siteNav) siteNav.classList.toggle("scrolled", window.scrollY > 18);
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", () => {
    updateStaticIndicators();
    setupHeroMarquee();
    if (window.innerWidth > 760) closeMobileNav();
  });

  onScroll();
  moveIndicator(journeyIndicator, document.querySelector(".rail button.active"));
  moveIndicator(resumeIndicator, document.querySelector(".resume-tabs button.active"));
  updateStaticIndicators();
  setupHeroMarquee();
})();
