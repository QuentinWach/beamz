(() => {
  const root = document.documentElement;
  let navigationFallback = null;

  const nextFrame = () =>
    new Promise((resolve) => requestAnimationFrame(resolve));

  const beginDocsNavigation = () => {
    window.clearTimeout(navigationFallback);
    root.classList.add("beamz-docs-navigating");
    navigationFallback = window.setTimeout(() => {
      root.classList.remove("beamz-docs-navigating");
    }, 1800);
  };

  const finishDocsNavigation = async () => {
    const contentImages = Array.from(
      document.querySelectorAll(".md-content img"),
    ).filter((image) => !image.complete);
    const imagesReady = Promise.all(
      contentImages.map(
        (image) =>
          new Promise((resolve) => {
            image.addEventListener("load", resolve, { once: true });
            image.addEventListener("error", resolve, { once: true });
          }),
      ),
    );

    try {
      await Promise.all([
        document.fonts?.ready || Promise.resolve(),
        Promise.race([
          imagesReady,
          new Promise((resolve) => window.setTimeout(resolve, 1200)),
        ]),
      ]);
    } catch (_) {
      // Asset failures should not prevent the new page from being revealed.
    }

    await nextFrame();
    await nextFrame();
    window.clearTimeout(navigationFallback);
    root.classList.remove("beamz-docs-navigating");
  };

  document.addEventListener(
    "click",
    (event) => {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }

      const link = event.target.closest?.("a[href]");
      if (!link || link.target || link.hasAttribute("download")) return;

      const destination = new URL(link.href, window.location.href);
      const docsRoot = "/docs/";
      const isCurrentPage =
        destination.pathname === window.location.pathname &&
        destination.search === window.location.search;

      if (
        destination.origin === window.location.origin &&
        destination.pathname.startsWith(docsRoot) &&
        !isCurrentPage
      ) {
        beginDocsNavigation();
      }
    },
    true,
  );

  window.addEventListener("popstate", beginDocsNavigation);

  const initializeHeader = () => {
    const header = document.querySelector(".beamz-site-header");
    if (!header) return;

    const themeToggle = header.querySelector(".beamz-site-header__theme");
    const updateThemeToggle = (scheme) => {
      if (!themeToggle) return;
      const isDark = scheme === "slate";
      themeToggle.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode" : "Switch to dark mode",
      );
      themeToggle.setAttribute("aria-pressed", String(isDark));
    };

    const savedTheme = localStorage.getItem("beamz-theme");
    const theme = savedTheme === "dark" ? "dark" : "light";
    const scheme = theme === "dark" ? "slate" : "default";
    document.documentElement.dataset.theme = theme;
    document.body.dataset.mdColorScheme = scheme;
    updateThemeToggle(scheme);

    if (themeToggle && !themeToggle.dataset.listenerAttached) {
      themeToggle.addEventListener("click", () => {
        const nextScheme = document.body.dataset.mdColorScheme === "slate" ? "default" : "slate";
        const nextTheme = nextScheme === "slate" ? "dark" : "light";
        document.documentElement.dataset.theme = nextTheme;
        document.body.dataset.mdColorScheme = nextScheme;
        localStorage.setItem("beamz-theme", nextTheme);
        updateThemeToggle(nextScheme);
      });
      themeToggle.dataset.listenerAttached = "true";
    }

    if (!header.dataset.scrollListenerAttached) {
      const updateScrolled = () => {
        header.classList.toggle(
          "beamz-site-header--scrolled",
          window.scrollY > 8,
        );
      };

      updateScrolled();
      window.addEventListener("scroll", updateScrolled, { passive: true });
      header.dataset.scrollListenerAttached = "true";
    }

    if (header.dataset.githubMetadataLoaded) return;
    header.dataset.githubMetadataLoaded = "true";

    fetch("https://api.github.com/repos/quentinwach/beamz")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        const stars = document.getElementById("beamz-doc-stars");
        if (stars && data?.stargazers_count != null) {
          stars.textContent = String(data.stargazers_count);
        }
      })
      .catch(() => {});

    fetch("https://api.github.com/repos/quentinwach/beamz/releases/latest")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        const version = document.getElementById("beamz-doc-version");
        if (version && data?.tag_name) version.textContent = data.tag_name;
      })
      .catch(() => {});
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => {
      initializeHeader();
      finishDocsNavigation();
    });
  } else if (document.readyState === "loading") {
    document.addEventListener(
      "DOMContentLoaded",
      () => {
        initializeHeader();
        finishDocsNavigation();
      },
      { once: true },
    );
  } else {
    initializeHeader();
    finishDocsNavigation();
  }
})();
