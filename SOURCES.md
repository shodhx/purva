# Sources

## khabarbhojpuri (news)
- URL: https://khabarbhojpuri.com
- Access: static HTML, category pages
- robots.txt: checked 2026-06-22; disallows only /?s=, /search/, /wp-json/, /?rest_route=. Category and article pages allowed.
- Language: Bhojpuri (Devanagari), manually verified on article bodies.
- Pilot run 2026-06-22: 25 articles, 527 raw sentences, 500 kept after cleaning and dedup.

## Bhojpuri Wikipedia (encyclopedic)
- URL: https://bh.wikipedia.org (MediaWiki Action API)
- License: CC BY-SA (redistributable)
- Method: random article sampling via API, plain-text extracts
- Filtering: calendar/date-page boilerplate, wiki markup, navigation fragments removed
- Collected 2026-06-26: 2,431 sentences

## Bhojpuri Sahitya Sarita (literary)
- URL: http://www.bhojpurisahityasarita.com
- robots.txt: checked 2026-07-04; category crawl permitted
- Method: WordPress collector, category auto-discovery via sitemap, 15 leaf categories
- Registers: essays, poetry, songs, ghazals, stories, flash fiction, reviews, memoirs, editorials, satire, interviews
- Collected 2026-07-04: 3,819 sentences (300/category cap; 4 categories archive-exhausted below cap)

## Jogira (literary portal)
- URL: https://jogira.com
- robots.txt: checked 2026-07-05; permitted
- Method: sitemap-walk (16 post-sitemaps, ~3,200 posts), slug include/exclude filters
- Kept: kahani/kavita/gazal/geet/katha/proverbs/laghu slugs; excluded wallpaper/film/actor content
- Collected 2026-07-05: 7,337 sentences

## Anjoria — assessed, excluded
- URL: https://anjoria.com
- Largest and longest-running Bhojpuri portal (archive to 2011); genuine Bhojpuri content confirmed manually
- Site employs bot protection returning 403 to all scripted requests (including robots.txt); collection not pursued

## Kavita Kosh (canonical verse)
- URL: https://kavitakosh.org (MediaWiki core revisions API)
- License: content copyrighted to named poets; pointer-only release
- Method: category-tree walk from श्रेणी:भोजपुरी रचना and श्रेणी:भोजपुरी गीत
- Note: गीत is largely a subset of रचना (only 18 new sentences beyond रचना)
- Collected 2026-07-05: 3,377 (रचना) + 18 new (गीत) = 3,395 sentences
- Excluded: भोजपुरी रचनाकार (poet bios), भोजपुरी भाषा (meta-content about the language)

## Assessed and excluded
- YouTube comments: manually assessed; genuine Bhojpuri <0.1% of comments on Bhojpuri-content videos; excluded as impractical
- Reddit (r/Bhojpuriyas): public JSON endpoints return 403 (IP-level block) despite compliant User-Agent; collector retained in repo
- FLEURS (google/fleurs): no Bhojpuri configuration exists
- ai4bharat/Rural_Women_Bhojpuri: gated (401 Unauthorized); access request required