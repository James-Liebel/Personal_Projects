(() => {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const preloader = document.getElementById("preloader");
  const preloaderBarFill = document.getElementById("preloaderBarFill");
  const preloaderMark = preloader?.querySelector(".preloader-mark") ?? null;
  const preloaderTop = preloader?.querySelector(".preloader-panel-top") ?? null;
  const preloaderBottom = preloader?.querySelector(".preloader-panel-bottom") ?? null;
  const siteNav = document.getElementById("siteNav");
  const scrollProgress = document.getElementById("scroll-progress");
  const brandBadges = [...document.querySelectorAll(".brand-badge")];
  const navLinks = [...document.querySelectorAll('.nav-link[href^="#"]')];
  const sectionTargets = [...document.querySelectorAll("main section[id]")];
  const navIndicator = document.getElementById("navIndicator");
  const navToggle = document.getElementById("navToggle");
  const navClose = document.getElementById("navClose");
  const mobileNav = document.getElementById("mobileNav");
  const customCursor = document.getElementById("customCursor");
  const customCursorDot = document.getElementById("customCursorDot");
  const hero = document.getElementById("hero");
  const heroTitle = document.getElementById("heroTitle");
  const heroTerminal = document.getElementById("heroTerminal");
  const heroPills = document.getElementById("heroPills");
  const heroGrain = hero?.querySelector(".hero-grain") ?? null;
  const heroWord = document.getElementById("heroWord");
  const modeButtons = [...document.querySelectorAll(".switch button[data-mode]")];
  const modeIndicator = document.getElementById("modeIndicator");
  const signalTags = document.getElementById("signalTags");
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
  const projectLinks = [...document.querySelectorAll(".project-rail a[href^='#project-']")];
  const projectSections = [...document.querySelectorAll(".project-section[data-project-section]")];
  const journey = document.getElementById("journey");
  const journeyTrack = document.getElementById("journey-track");
  const journeyLinks = [...document.querySelectorAll(".rail a[data-journey]")];
  const journeyStops = [...document.querySelectorAll(".journey-stop[data-panel]")];
  const resumePanel = document.querySelector(".resume-panel");
  const footer = document.querySelector(".site-footer");
  const pdfButtons = [document.getElementById("pdfBtn")].filter(Boolean);

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
      s2: ["Evaluation", "Validation / benchmarks / tradeoffs", "Results are framed with metrics, comparisons, and tradeoffs."],
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

  const heroWords = ["data science", "machine learning", "visualization", "analytics"];
  let heroWordIndex = 0;
  let lenis = null;
  let gsapScrollProgress = false;
  let heroWordTimer = null;
  let heroTypeTimer = null;
  let heroMotionStarted = false;
  let journeyScrollTrigger = null;

  function animateCountUpItem(item, duration = 1.4) {
    if (!item || item.dataset.counted === "true" || !window.gsap) return;
    const { gsap } = window;
    const target = Number(item.dataset.value || "0");
    const suffix = item.dataset.suffix || "";
    const state = { value: 0 };
    const formatValue = value => `${Math.round(value).toLocaleString()}${suffix}`;
    item.dataset.counted = "true";
    item.textContent = formatValue(0);
    gsap.to(state, {
      value: target,
      duration,
      ease: "power2.out",
      onUpdate: () => {
        item.textContent = formatValue(state.value);
      }
    });
  }

  function movePillIndicator(indicator, target) {
    if (!indicator || !target || !target.parentElement) return;
    const parentRect = target.parentElement.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    indicator.style.width = `${rect.width}px`;
    indicator.style.transform = `translate(${rect.left - parentRect.left}px, 0)`;
  }

  function moveNavIndicator(activeLink) {
    if (!navIndicator || !activeLink || !activeLink.parentElement || window.innerWidth <= 920) return;
    const parentRect = activeLink.parentElement.getBoundingClientRect();
    const rect = activeLink.getBoundingClientRect();
    navIndicator.style.width = `${rect.width}px`;
    navIndicator.style.transform = `translate(${rect.left - parentRect.left}px, 0)`;
    navIndicator.classList.add("ready");
  }

  function getNavLinkBaseColor(link) {
    if (!link) return "";
    const scrolled = siteNav?.classList.contains("scrolled");
    if (scrolled) {
      return link.classList.contains("active") ? "var(--dark)" : "rgba(36, 50, 79, 0.68)";
    }
    return link.classList.contains("active") ? "var(--cream)" : "rgba(255, 247, 239, 0.84)";
  }

  function applyNavLinkState(link) {
    if (!link) return;
    link.style.color = getNavLinkBaseColor(link);
    link.style.transform = link.classList.contains("active") ? "translateY(-2px)" : "translateY(0)";
  }

  function syncNavLinkStyles() {
    navLinks.forEach(applyNavLinkState);
  }

  function markActiveNav(id) {
    let activeLink = null;
    navLinks.forEach(link => {
      const isActive = link.getAttribute("href") === `#${id}`;
      link.classList.toggle("active", isActive);
      if (isActive) activeLink = link;
    });
    syncNavLinkStyles();
    moveNavIndicator(activeLink);
  }

  function closeMobileNav() {
    if (!mobileNav || mobileNav.hidden) return;
    mobileNav.hidden = true;
    document.body.classList.remove("menu-open");
    if (navToggle) navToggle.setAttribute("aria-expanded", "false");
  }

  function openMobileNav() {
    if (!mobileNav) return;
    mobileNav.hidden = false;
    document.body.classList.add("menu-open");
    if (navToggle) navToggle.setAttribute("aria-expanded", "true");
  }

  function scrollToTarget(target) {
    if (!target) return;
    if (lenis) {
      lenis.scrollTo(target, { offset: -88, duration: 1.05 });
    } else {
      target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    }
  }

  function updateProgress() {
    if (gsapScrollProgress) return;
    if (!scrollProgress) return;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const progress = max > 0 ? (window.scrollY / max) * 100 : 0;
    scrollProgress.style.width = `${progress}%`;
  }

  function updateNavState() {
    if (!siteNav || !hero) return;
    const heroBottom = hero.getBoundingClientRect().bottom;
    siteNav.classList.toggle("scrolled", heroBottom <= 140);
    syncNavLinkStyles();
  }

  function splitHeadlineWords() {
    if (!heroTitle || heroTitle.dataset.split === "true") {
      return heroTitle ? [...heroTitle.querySelectorAll(".hero-word-fragment")] : [];
    }
    const lineNodes = [...heroTitle.querySelectorAll(".hero-name, .hero-line")].filter(node => !node.classList.contains("hero-rotator"));
    lineNodes.forEach(line => {
      const text = line.textContent || "";
      const words = text.trim().split(/\s+/).filter(Boolean);
      if (!words.length) return;
      line.textContent = "";
      words.forEach((word, index) => {
        const fragment = document.createElement("span");
        fragment.className = "hero-word-fragment";
        fragment.textContent = word;
        line.appendChild(fragment);
        if (index < words.length - 1) {
          line.appendChild(document.createTextNode(" "));
        }
      });
    });
    heroTitle.dataset.split = "true";
    return [...heroTitle.querySelectorAll(".hero-word-fragment")];
  }

  function renderMode(key) {
    const next = modeData[key];
    if (!next) return;
    const activeButton = modeButtons.find(button => button.dataset.mode === key);
    modeButtons.forEach(button => button.classList.toggle("active", button === activeButton));
    movePillIndicator(modeIndicator, activeButton);
    if (ring) ring.style.setProperty("--angle", `${next.angle}deg`);
    if (ringScore) ringScore.textContent = next.word;
    if (ringLabel) ringLabel.textContent = next.label;
    if (ringFoot) ringFoot.textContent = next.foot;
    if (signalTitle) signalTitle.textContent = next.title;
    if (signalBadge) signalBadge.textContent = next.badge;
    if (signalBody) signalBody.textContent = next.body;
    [[s1l, s1v, s1n, next.s1], [s2l, s2v, s2n, next.s2], [s3l, s3v, s3n, next.s3]].forEach(([a, b, c, values]) => {
      if (a) a.textContent = values[0];
      if (b) b.textContent = values[1];
      if (c) c.textContent = values[2];
    });
    if (signalTags) {
      signalTags.innerHTML = next.tags.map(tag => `<span class="tag">${tag}</span>`).join("");
    }
  }

  function updateProjectRail(activeId) {
    projectLinks.forEach(link => {
      const isActive = link.getAttribute("href") === `#${activeId}`;
      link.classList.toggle("active", isActive);
    });
  }

  function updateJourneyRail(activePanel) {
    journeyLinks.forEach(link => {
      const isActive = link.dataset.journey === activePanel;
      link.classList.toggle("active", isActive);
    });
  }

  function setupObservers() {
    const navObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) markActiveNav(entry.target.id);
      });
    }, { rootMargin: "-20% 0px -65% 0px", threshold: 0.15 });

    sectionTargets.forEach(section => navObserver.observe(section));

    const projectObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        updateProjectRail(entry.target.id);
      });
    }, { threshold: 0.45 });

    projectSections.forEach(section => projectObserver.observe(section));

    if (reduced) {
      const journeyObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (!entry.isIntersecting) return;
          updateJourneyRail(entry.target.dataset.panel || "");
        });
      }, { root: document.querySelector(".journey-track-shell"), threshold: 0.55 });

      journeyStops.forEach(stop => journeyObserver.observe(stop));
    }
  }

  function setupCountUps() {
    const items = [...document.querySelectorAll(".count-up")];
    if (!items.length || reduced || !window.gsap || !window.ScrollTrigger) return;
    const { ScrollTrigger } = window;

    items.forEach(item => {
      ScrollTrigger.create({
        trigger: item,
        start: "top 82%",
        once: true,
        onEnter: () => {
          animateCountUpItem(item);
        }
      });
    });
  }

  function setupBentoMotion() {
    if (reduced || !window.gsap || !window.ScrollTrigger) return;
    const { gsap } = window;
    const grid = document.querySelector(".skills-grid");
    const cards = grid ? [...grid.querySelectorAll(".bento-card")] : [];
    if (!grid || !cards.length) return;

    gsap.from(cards, {
      y: 60,
      opacity: 0,
      stagger: 0.09,
      duration: 0.65,
      ease: "power3.out",
      scrollTrigger: {
        trigger: grid,
        start: "top 80%",
        once: true
      }
    });

    if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

    cards.forEach(card => {
      const state = {
        x: 0,
        y: 0,
        targetX: 0,
        targetY: 0,
        raf: 0
      };

      const render = () => {
        state.x += (state.targetX - state.x) * 0.1;
        state.y += (state.targetY - state.y) * 0.1;
        card.style.setProperty("--magnetic-x", `${state.x.toFixed(2)}px`);
        card.style.setProperty("--magnetic-y", `${state.y.toFixed(2)}px`);

        const settled = Math.abs(state.targetX - state.x) < 0.08 && Math.abs(state.targetY - state.y) < 0.08;
        if (settled && state.targetX === 0 && state.targetY === 0) {
          state.raf = 0;
          return;
        }
        state.raf = requestAnimationFrame(render);
      };

      const startRender = () => {
        if (!state.raf) state.raf = requestAnimationFrame(render);
      };

      card.addEventListener("mousemove", event => {
        gsap.killTweensOf(state);
        const rect = card.getBoundingClientRect();
        const offsetX = ((event.clientX - rect.left) / rect.width - 0.5) * 16;
        const offsetY = ((event.clientY - rect.top) / rect.height - 0.5) * 16;
        state.targetX = Math.max(-8, Math.min(8, offsetX));
        state.targetY = Math.max(-8, Math.min(8, offsetY));
        startRender();
      });

      card.addEventListener("mouseleave", () => {
        if (state.raf) {
          cancelAnimationFrame(state.raf);
          state.raf = 0;
        }
        gsap.to(state, {
          x: 0,
          y: 0,
          targetX: 0,
          targetY: 0,
          duration: 0.9,
          ease: "elastic.out(1, 0.45)",
          overwrite: true,
          onUpdate: () => {
            card.style.setProperty("--magnetic-x", `${state.x.toFixed(2)}px`);
            card.style.setProperty("--magnetic-y", `${state.y.toFixed(2)}px`);
          }
        });
      });
    });
  }

  function setupProjectShowcase() {
    if (reduced || !window.gsap || !window.ScrollTrigger) return;
    const { gsap, ScrollTrigger } = window;

    const fraudGrid = document.querySelector(".fraud-grid");
    if (fraudGrid) {
      fraudGrid.innerHTML = "";
      const cells = [];
      for (let index = 0; index < 240; index += 1) {
        const cell = document.createElement("span");
        cell.className = "fraud-cell";
        fraudGrid.appendChild(cell);
        cells.push(cell);
      }

      const flashFraudCells = () => {
        const flashes = 3 + Math.floor(Math.random() * 3);
        const activeIndexes = new Set();
        while (activeIndexes.size < flashes) {
          activeIndexes.add(Math.floor(Math.random() * cells.length));
        }

        activeIndexes.forEach(index => {
          const cell = cells[index];
          if (!cell) return;
          cell.classList.add("is-hot");
          window.setTimeout(() => cell.classList.remove("is-hot"), 260 + Math.random() * 180);
        });
      };

      flashFraudCells();
      window.setInterval(flashFraudCells, 200);
    }

    const chatMessages = document.querySelector(".chat-messages");
    if (chatMessages) {
      const chatLines = [
        { text: "Running fraud model...", alt: false },
        { text: "Benchmark check complete ✓", alt: true },
        { text: "Anomalies flagged: 1,247", alt: false }
      ];
      const chatTimers = [];
      let chatStarted = false;

      const clearChatTimers = () => {
        while (chatTimers.length) {
          window.clearTimeout(chatTimers.pop());
        }
      };

      const createTypingIndicator = isAlt => {
        const typing = document.createElement("div");
        typing.className = `chat-typing${isAlt ? " alt" : ""}`;
        typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
        return typing;
      };

      const runChatLoop = () => {
        clearChatTimers();
        chatMessages.innerHTML = "";
        let cursor = 0;

        chatLines.forEach(line => {
          const typing = createTypingIndicator(line.alt);

          chatTimers.push(window.setTimeout(() => {
            chatMessages.appendChild(typing);
            gsap.fromTo(typing, { y: 10, opacity: 0 }, { y: 0, opacity: 1, duration: 0.25, ease: "power2.out" });
          }, cursor));

          chatTimers.push(window.setTimeout(() => {
            const message = document.createElement("div");
            message.className = `chat-message${line.alt ? " alt" : ""}`;
            message.textContent = line.text;
            typing.replaceWith(message);
            gsap.fromTo(message, { y: 10, opacity: 0 }, { y: 0, opacity: 1, duration: 0.3, ease: "power2.out" });
          }, cursor + 560));

          cursor += 1040;
        });

        chatTimers.push(window.setTimeout(runChatLoop, cursor + 4000));
      };

      ScrollTrigger.create({
        trigger: chatMessages.closest(".project-section"),
        start: "top 68%",
        once: true,
        onEnter: () => {
          if (chatStarted) return;
          chatStarted = true;
          runChatLoop();
        }
      });
    }

    const newsTrack = document.querySelector(".news-track");
    if (newsTrack && newsTrack.dataset.duplicated !== "true") {
      [...newsTrack.children].forEach(item => {
        newsTrack.appendChild(item.cloneNode(true));
      });
      newsTrack.dataset.duplicated = "true";
    }

    projectSections.forEach(section => {
      const inner = section.querySelector(".project-inner");
      const copy = section.querySelector(".project-copy");
      const visual = section.querySelector(".work-visual-card");
      if (!copy || !visual) return;

      const isReverse = inner?.classList.contains("reverse");
      const copyOffset = isReverse ? 60 : -60;
      const visualOffset = isReverse ? -60 : 60;

      gsap.set(copy, { x: copyOffset, opacity: 0 });
      gsap.set(visual, { x: visualOffset, opacity: 0 });

      gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: "top 74%",
          once: true
        }
      })
        .to(copy, {
          x: 0,
          opacity: 1,
          duration: 0.8,
          ease: "power3.out"
        })
        .to(visual, {
          x: 0,
          opacity: 1,
          duration: 0.8,
          ease: "power3.out"
        }, 0.08);

      gsap.to(visual, {
        "--float-y": "-12px",
        duration: 4,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true
      });

      ScrollTrigger.create({
        trigger: section,
        start: "top center",
        end: "bottom center",
        onEnter: () => updateProjectRail(section.id),
        onEnterBack: () => updateProjectRail(section.id)
      });
    });
  }

  function setupVisualizationGallery() {
    if (reduced || !window.gsap || !window.ScrollTrigger) return;
    const { gsap } = window;
    const section = document.getElementById("visualizations");
    const heading = section?.querySelector(".section-head h2") ?? null;
    const grid = section?.querySelector(".visual-grid") ?? null;
    const cards = grid ? [...grid.querySelectorAll(".visual-card")] : [];
    if (!section || !heading || !grid || !cards.length) return;

    if (heading.dataset.split !== "true") {
      const chars = [...(heading.textContent || "")];
      heading.textContent = "";
      chars.forEach(char => {
        const span = document.createElement("span");
        span.className = "visual-heading-char";
        span.textContent = char === " " ? "\u00A0" : char;
        heading.appendChild(span);
      });
      heading.dataset.split = "true";
    }

    const headingChars = [...heading.querySelectorAll(".visual-heading-char")];
    gsap.from(headingChars, {
      y: 30,
      opacity: 0,
      stagger: 0.025,
      duration: 0.45,
      ease: "power3.out",
      scrollTrigger: {
        trigger: section,
        start: "top 74%",
        once: true
      }
    });

    cards.forEach(card => {
      const frame = card.querySelector(".visual-frame");
      if (frame && !frame.querySelector(".visual-overlay")) {
        const titleSource = card.querySelector("h3, .visual-label, .powerbi-embed-meta strong");
        const actionSource = card.querySelector(".visual-actions a");
        const overlay = document.createElement("div");
        const copy = document.createElement("div");
        const title = document.createElement("strong");
        const action = document.createElement("span");
        overlay.className = "visual-overlay";
        overlay.setAttribute("aria-hidden", "true");
        copy.className = "visual-overlay-copy";
        title.textContent = titleSource ? titleSource.textContent.trim() : "";
        action.textContent = actionSource ? actionSource.textContent.trim() : "";
        copy.append(title, action);
        overlay.appendChild(copy);
        frame.appendChild(overlay);
      }
    });

    gsap.from(cards, {
      y: 55,
      opacity: 0,
      stagger: 0.09,
      duration: 0.65,
      ease: "power3.out",
      scrollTrigger: {
        trigger: grid,
        start: "top 82%",
        once: true
      }
    });

    cards.forEach(card => {
      const line = card.querySelector(".visual-line");
      if (!line) return;
      gsap.set(line, { scaleX: 0 });
      gsap.to(line, {
        scaleX: 1,
        duration: 0.6,
        ease: "power2.out",
        scrollTrigger: {
          trigger: card,
          start: "top 85%",
          once: true
        }
      });
    });
  }

  function scrollToJourneyStop(target) {
    if (!target || !journeyScrollTrigger || !journeyTrack) {
      scrollToTarget(target);
      return;
    }

    const maxX = Math.max(1, journeyTrack.scrollWidth - window.innerWidth);
    const ratio = Math.min(1, Math.max(0, target.offsetLeft / maxX));
    const destination = journeyScrollTrigger.start + (journeyScrollTrigger.end - journeyScrollTrigger.start) * ratio;

    if (lenis) {
      lenis.scrollTo(destination, { duration: 1.05 });
    } else {
      window.scrollTo({ top: destination, behavior: reduced ? "auto" : "smooth" });
    }
  }

  function setupJourneyMotion() {
    if (reduced || !window.gsap || !window.ScrollTrigger || !journey || !journeyTrack || !journeyStops.length) return;
    const { gsap, ScrollTrigger } = window;
    journey.classList.add("journey-motion-active");

    const maxOffset = () => Math.max(0, journeyTrack.scrollWidth - window.innerWidth);
    const journeyTween = gsap.to(journeyTrack, {
      x: () => -maxOffset(),
      ease: "none",
      scrollTrigger: {
        trigger: "#journey",
        pin: true,
        scrub: 1,
        anticipatePin: 1,
        invalidateOnRefresh: true,
        end: () => `+=${Math.max(maxOffset(), window.innerWidth)}`
      }
    });

    journeyScrollTrigger = journeyTween.scrollTrigger;

    journeyStops.forEach(stop => {
      ScrollTrigger.create({
        trigger: stop,
        containerAnimation: journeyTween,
        start: "left center",
        end: "right center",
        onEnter: () => updateJourneyRail(stop.dataset.panel || ""),
        onEnterBack: () => updateJourneyRail(stop.dataset.panel || "")
      });
    });

    const overview = document.getElementById("journey-overview");
    const overviewCounts = overview ? [...overview.querySelectorAll(".count-up")] : [];
    if (overview && overviewCounts.length) {
      ScrollTrigger.create({
        trigger: overview,
        containerAnimation: journeyTween,
        start: "center center",
        once: true,
        onEnter: () => {
          overviewCounts.forEach(item => animateCountUpItem(item));
        }
      });
    }

    document.querySelectorAll(".journey-node").forEach(node => {
      const dot = node.querySelector(".journey-node-dot");
      const line = node.querySelector(".journey-node-line");
      const stop = node.closest(".journey-stop");
      if (!dot || !line || !stop) return;

      gsap.set(dot, { scale: 0.72, opacity: 0.7 });
      gsap.set(line, { scaleY: 0 });

      gsap.timeline({
        scrollTrigger: {
          trigger: stop,
          containerAnimation: journeyTween,
          start: "left 64%",
          once: true
        }
      })
        .to(dot, {
          scale: 1,
          opacity: 1,
          duration: 0.3,
          ease: "power2.out"
        })
        .to(line, {
          scaleY: 1,
          duration: 0.45,
          ease: "power2.out"
        }, "-=0.05");
    });

    const nextCard = document.querySelector(".journey-next-card");
    if (nextCard) {
      const pulse = gsap.to(nextCard, {
        opacity: 1,
        duration: 2,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true,
        paused: true
      });

      gsap.from(nextCard, {
        x: 80,
        opacity: 0,
        duration: 0.8,
        ease: "power3.out",
        scrollTrigger: {
          trigger: nextCard,
          containerAnimation: journeyTween,
          start: "left 76%",
          once: true,
          onEnter: () => pulse.play()
        }
      });
    }
  }

  function setupMicroInteractions() {
    if (reduced || !window.gsap || !window.ScrollTrigger) return;
    const { gsap, ScrollTrigger } = window;

    const pressables = [
      ...document.querySelectorAll("button"),
      ...document.querySelectorAll(".btn, .btn-soft, .signal-link, .project-open, .resume-link, .nav-icon-link")
    ];
    const uniquePressables = [...new Set(pressables)];

    uniquePressables.forEach(element => {
      const press = () => {
        gsap.to(element, { scale: 0.95, duration: 0.12, ease: "power2.out", overwrite: true });
      };
      const release = () => {
        gsap.to(element, { scale: 1, duration: 0.38, ease: "back.out(2)", overwrite: true });
      };

      element.addEventListener("pointerdown", press);
      element.addEventListener("pointerup", release);
      element.addEventListener("pointerleave", release);
      element.addEventListener("pointercancel", release);
    });

    navLinks.forEach(link => {
      link.addEventListener("mouseenter", () => {
        gsap.to(link, { y: -2, color: "var(--orange)", duration: 0.15, overwrite: true });
      });
      link.addEventListener("mouseleave", () => {
        gsap.to(link, {
          y: link.classList.contains("active") ? -2 : 0,
          color: getNavLinkBaseColor(link),
          duration: 0.15,
          overwrite: true
        });
      });
    });

    brandBadges.forEach(badge => {
      badge.addEventListener("mouseenter", () => {
        gsap.to(badge, { rotate: "+=360", duration: 0.55, ease: "power2.inOut", overwrite: true });
      });
    });

    document.querySelectorAll(".nav-availability-dot").forEach(dot => {
      gsap.to(dot, {
        scale: 1.4,
        opacity: 0.4,
        duration: 0.75,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true
      });
    });

    const eyebrowTargets = [
      ...document.querySelectorAll(".section-head .section-kicker"),
      ...document.querySelectorAll(".work-intro .section-kicker"),
      ...document.querySelectorAll(".journey-head .section-kicker"),
      ...document.querySelectorAll(".resume-kicker")
    ];

    [...new Set(eyebrowTargets)].forEach(label => {
      gsap.from(label, {
        letterSpacing: "0.05em",
        opacity: 0,
        duration: 0.55,
        ease: "power2.out",
        scrollTrigger: {
          trigger: label,
          start: "top 88%",
          once: true
        }
      });
    });

    if (resumePanel) {
      gsap.from(resumePanel, {
        scale: 0.96,
        opacity: 0,
        duration: 0.6,
        ease: "power3.out",
        scrollTrigger: {
          trigger: resumePanel,
          start: "top 82%",
          once: true
        }
      });
    }

    if (footer) {
      const footerColumns = [...footer.querySelectorAll(".footer-column")];
      const footerBottomGroup = [footer.querySelector(".footer-divider"), footer.querySelector(".footer-bottom")].filter(Boolean);

      if (footerColumns.length) {
        gsap.from(footerColumns, {
          y: 28,
          opacity: 0,
          duration: 0.55,
          stagger: 0.08,
          ease: "power3.out",
          scrollTrigger: {
            trigger: footer,
            start: "top 82%",
            once: true
          }
        });
      }

      if (footerBottomGroup.length) {
        gsap.from(footerBottomGroup, {
          y: 24,
          opacity: 0,
          duration: 0.45,
          stagger: 0.08,
          ease: "power2.out",
          scrollTrigger: {
            trigger: footer,
            start: "top 76%",
            once: true
          }
        });
      }
    }
  }

  function typeSignalBody() {
    if (!signalBody) return;
    const fullText = signalBody.textContent || "";
    if (!fullText) return;
    signalBody.textContent = "";
    if (heroTypeTimer) window.clearTimeout(heroTypeTimer);
    let index = 0;
    const step = () => {
      signalBody.textContent = fullText.slice(0, index);
      index += 1;
      if (index <= fullText.length) {
        heroTypeTimer = window.setTimeout(step, 16);
      }
    };
    step();
  }

  function setupHeroWordCycle() {
    if (!heroWord || !window.gsap) return;
    if (heroWordTimer) window.clearInterval(heroWordTimer);
    heroWord.style.opacity = "1";
    heroWord.style.transform = "translateY(0%)";
    heroWordTimer = window.setInterval(() => {
      heroWordIndex = (heroWordIndex + 1) % heroWords.length;
      const nextWord = heroWords[heroWordIndex];
      const tl = window.gsap.timeline();
      tl.to(heroWord, {
        yPercent: -100,
        opacity: 0,
        duration: 0.4,
        ease: "power2.inOut"
      })
        .call(() => {
          heroWord.textContent = nextWord;
          window.gsap.set(heroWord, { yPercent: 100 });
        })
        .to(heroWord, {
          yPercent: 0,
          opacity: 1,
          duration: 0.4,
          ease: "power2.out"
        }, "+=0.1");
    }, 3000);
  }

  function setupHeroParallax() {
    if (!heroTitle || !window.gsap) return;
    document.addEventListener("mousemove", event => {
      const dx = (event.clientX / window.innerWidth - 0.5) * 16;
      const dy = (event.clientY / window.innerHeight - 0.5) * 8;
      window.gsap.to("[data-headline]", { x: dx, y: dy, duration: 0.9, ease: "power2.out", overwrite: true });
    }, { passive: true });
  }

  function setupHeroGrain() {
    if (!heroGrain) return;
    const context = heroGrain.getContext("2d");
    if (!context) return;

    const resize = () => {
      const rect = heroGrain.getBoundingClientRect();
      heroGrain.width = Math.max(1, Math.floor(rect.width));
      heroGrain.height = Math.max(1, Math.floor(rect.height));
    };

    const render = () => {
      const { width, height } = heroGrain;
      context.clearRect(0, 0, width, height);
      for (let i = 0; i < 1200; i += 1) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const alpha = Math.random() * 0.18;
        context.fillStyle = `rgba(250,247,242,${alpha})`;
        context.fillRect(x, y, 1, 1);
      }
      requestAnimationFrame(render);
    };

    resize();
    window.addEventListener("resize", resize);
    requestAnimationFrame(render);
  }

  function startHeroMotion() {
    if (heroMotionStarted || !window.gsap) return;
    heroMotionStarted = true;
    setupHeroGrain();
    setupHeroParallax();

    const headlineWords = splitHeadlineWords();
    const pillItems = heroPills ? [...heroPills.children] : [];
    const tl = window.gsap.timeline();

    window.gsap.set(headlineWords, { y: 40, opacity: 0 });
    if (heroTerminal) window.gsap.set(heroTerminal, { x: 80, opacity: 0 });
    window.gsap.set(pillItems, { y: 20, opacity: 0 });

    tl.to(headlineWords, {
      y: 0,
      opacity: 1,
      duration: 0.7,
      stagger: 0.06,
      ease: "power3.out"
    })
      .to(heroTerminal, {
        x: 0,
        opacity: 1,
        duration: 0.8,
        ease: "power3.out",
        onComplete: typeSignalBody
      }, "-=0.18")
      .to(pillItems, {
        y: 0,
        opacity: 1,
        duration: 0.3,
        stagger: 0.05,
        ease: "power2.out"
      }, "-=0.2");

    setupHeroWordCycle();
  }

  function setupVizFallbacks() {
    document.querySelectorAll(".visual-frame").forEach(frameShell => {
      const frame = frameShell.querySelector("iframe");
      const fallback = frameShell.querySelector(".viz-fallback");
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
      frame.addEventListener("load", pass, { once: true });
      frame.addEventListener("error", fail, { once: true });
      window.setTimeout(() => {
        if (!settled) fail();
      }, 3500);
    });
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
    if (height > printableHeightPx) scale = Math.max(0.78, printableHeightPx / height);
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

  function setupLenis() {
    if (!window.Lenis || !window.gsap || !window.ScrollTrigger) return;
    const { gsap, ScrollTrigger, Lenis } = window;
    gsap.registerPlugin(ScrollTrigger);
    lenis = new Lenis({ lerp: 0.1, smoothWheel: true });
    lenis.on("scroll", ScrollTrigger.update);
    lenis.on("scroll", updateNavState);
    gsap.ticker.add(time => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);
    gsap.to("#scroll-progress", {
      width: "100%",
      ease: "none",
      scrollTrigger: {
        trigger: document.body,
        start: "top top",
        end: "bottom bottom",
        scrub: 0.3
      }
    });
    gsapScrollProgress = true;
  }

  function setupPreloader() {
    if (!preloader) return;
    if (!window.gsap) {
      preloader.classList.add("is-done");
      startHeroMotion();
      return;
    }
    const { gsap } = window;
    const finish = () => {
      preloader.classList.add("is-done");
      preloader.style.pointerEvents = "none";
      startHeroMotion();
    };

    window.addEventListener("load", () => {
      const tl = gsap.timeline({ onComplete: finish });
      gsap.set(preloader, { autoAlpha: 1 });
      gsap.set(preloaderBarFill, { scaleX: 0, transformOrigin: "left center" });
      gsap.set([preloaderTop, preloaderBottom], { yPercent: 0 });
      gsap.set(preloaderMark, { scale: 1, opacity: 1 });

      tl.to(preloaderBarFill, {
        scaleX: 1,
        duration: 1.8,
        ease: "power2.out"
      })
        .to(preloaderMark, {
          scale: 3.333,
          opacity: 0,
          duration: 0.45,
          ease: "power3.inOut"
        }, "-=0.05")
        .to(preloaderTop, {
          yPercent: -100,
          duration: 0.6,
          ease: "power3.inOut"
        }, "-=0.05")
        .to(preloaderBottom, {
          yPercent: 100,
          duration: 0.6,
          ease: "power3.inOut"
        }, "<")
        .to(preloader, {
          autoAlpha: 0,
          duration: 0.15
        }, "-=0.12");
    }, { once: true });
  }

  function setupCustomCursor() {
    if (!customCursor || !customCursorDot) return;
    if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches || window.innerWidth <= 920) return;
    const { gsap } = window;
    if (!gsap) return;

    document.body.classList.add("cursor-enabled");

    const state = {
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
      currentX: window.innerWidth / 2,
      currentY: window.innerHeight / 2,
      visible: false,
      linkHover: false,
      cardHover: false,
      darkSection: false
    };

    const updateHoverState = target => {
      state.linkHover = Boolean(target?.closest("a, button, [role='button']"));
      state.cardHover = Boolean(target?.closest(".bento-card, .visual-card, .work-visual-card, .journey-stop, .ring-card, .stack"));
      state.darkSection = Boolean(target?.closest("#hero, #visualizations, #journey"));
      customCursor.classList.toggle("is-hover-link", state.linkHover || state.cardHover);
      customCursor.classList.toggle("is-dark", state.darkSection);
    };

    const tick = () => {
      state.currentX += (state.x - state.currentX) * 0.12;
      state.currentY += (state.y - state.currentY) * 0.12;
      customCursor.style.transform = `translate3d(${state.currentX}px, ${state.currentY}px, 0) scale(${state.cardHover ? 3.5 : state.linkHover ? 2.5 : 1})`;
      customCursorDot.style.transform = `translate3d(${state.x}px, ${state.y}px, 0) scale(1)`;
      requestAnimationFrame(tick);
    };

    document.addEventListener("pointermove", event => {
      state.x = event.clientX;
      state.y = event.clientY;
      if (!state.visible) {
        state.visible = true;
        customCursor.classList.add("visible");
        customCursorDot.classList.add("visible");
      }
      updateHoverState(event.target);
    }, { passive: true });

    document.addEventListener("pointerleave", () => {
      state.visible = false;
      customCursor.classList.remove("visible");
      customCursorDot.classList.remove("visible");
    });

    window.addEventListener("blur", () => {
      customCursor.classList.remove("visible");
      customCursorDot.classList.remove("visible");
    });

    requestAnimationFrame(tick);
  }

  setupObservers();
  setupVizFallbacks();

  modeButtons.forEach(button => {
    button.addEventListener("click", () => renderMode(button.dataset.mode));
  });
  renderMode("builder");

  if (navToggle) navToggle.addEventListener("click", openMobileNav);
  if (navClose) navClose.addEventListener("click", closeMobileNav);
  if (mobileNav) {
    mobileNav.addEventListener("click", event => {
      if (event.target === mobileNav) closeMobileNav();
    });
  }

  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", event => {
      const target = document.querySelector(anchor.getAttribute("href"));
      if (!target) return;
      event.preventDefault();
      if (target.closest("#journey-track")) {
        scrollToJourneyStop(target);
      } else {
        scrollToTarget(target);
      }
      closeMobileNav();
    });
  });

  pdfButtons.forEach(button => button.addEventListener("click", printResume));
  window.printResume = printResume;

  window.addEventListener("resize", () => {
    movePillIndicator(modeIndicator, document.querySelector(".switch button.active"));
    moveNavIndicator(document.querySelector(".nav-link.active"));
    if (window.innerWidth > 920) closeMobileNav();
  });

  window.addEventListener("scroll", () => {
    if (!lenis) {
      updateProgress();
      updateNavState();
    }
  }, { passive: true });

  if (!reduced) {
    setupPreloader();
    setupLenis();
    setupBentoMotion();
    setupProjectShowcase();
    setupVisualizationGallery();
    setupJourneyMotion();
    setupCountUps();
    setupHeroWordCycle();
    setupCustomCursor();
    setupMicroInteractions();
  } else if (preloader) {
    preloader.classList.add("is-done");
  }

  updateProgress();
  updateNavState();
  markActiveNav("hero");
  updateProjectRail("project-fraud");
  updateJourneyRail("overview");
  movePillIndicator(modeIndicator, document.querySelector(".switch button.active"));
})();
